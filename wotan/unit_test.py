from wotan import flatten
import numpy
import time as ttime
from astropy.io import fits
#import matplotlib.pyplot as plt


def load_file(filename):
    """Loads a TESS *spoc* FITS file and returns TIME, PDCSAP_FLUX"""
    hdu = fits.open(filename)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']
    flux[flux == 0] = numpy.nan
    return time, flux


def main():
    print("Starting test: wotan synthetic...")
    numpy.random.seed(seed=0)  # reproducibility

    days = 10
    samples = 1000
    window = 1
    stdev = 0.01
    t = numpy.linspace(0, days, samples)
    data = numpy.random.normal(0, stdev, int(samples))

    # compile JIT numba
    print("Numba compilation...", end="")
    t1 = ttime.perf_counter()
    trend_lc = flatten(t, data, window, cval=6, ftol=1e-6)
    t2 = ttime.perf_counter()
    print("{0:.3f}".format(t2-t1), 'seconds')

    print('Detrending 1...', end="")
    t1 = ttime.perf_counter()
    flatten_lc, trend_lc = flatten(t, data, window, cval=6, ftol=1e-6, return_trend=True)
    t2 = ttime.perf_counter()
    print("{0:.3f}".format(t2-t1), 'seconds')

    numpy.testing.assert_equal(len(trend_lc), 1000)
    numpy.testing.assert_almost_equal(numpy.nanmax(trend_lc), 0.002154808411529983)
    numpy.testing.assert_almost_equal(numpy.nanmin(trend_lc), -0.0024286115432384527)
    numpy.testing.assert_almost_equal(trend_lc[500], -0.0006178575452455013)

    # TESS test
    filename = "https://archive.stsci.edu/hlsps/tess-data-alerts/" \
    "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
    time, flux = load_file(filename)
    flatten_lc, trend_lc = flatten(
        time,
        flux,
        window_length=0.5,
        edge_cutoff=1,
        break_tolerance=0.1,
        return_trend=True,
        cval=5.0)

    numpy.testing.assert_equal(len(trend_lc), 20076)
    numpy.testing.assert_almost_equal(numpy.nanmax(trend_lc), 28754.985299070882)
    numpy.testing.assert_almost_equal(numpy.nanmin(trend_lc), 28615.108124724477)
    numpy.testing.assert_almost_equal(trend_lc[500], 28671.686308143515)

    numpy.testing.assert_equal(len(flatten_lc), 20076)
    numpy.testing.assert_almost_equal(numpy.nanmax(flatten_lc), 1.0034653549250616)
    numpy.testing.assert_almost_equal(numpy.nanmin(flatten_lc), 0.996726610702177)
    numpy.testing.assert_almost_equal(flatten_lc[500], 1.000577429565131)

    """
    plt.scatter(time, flux, s=1, color='black')
    plt.plot(time, trend_lc, color='red')
    plt.show()
    plt.close()

    plt.scatter(time, flatten_lc, s=1, color='black')
    plt.show()
    """
    print('All tests completed.')


if __name__ == '__main__':
    main()
