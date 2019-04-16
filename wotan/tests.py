from __future__ import print_function
from wotan import flatten
import numpy
from astropy.io import fits


def load_file(filename):
    """Loads a TESS *spoc* FITS file and returns TIME, PDCSAP_FLUX"""
    hdu = fits.open(filename)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']
    flux[flux == 0] = numpy.nan
    return time, flux


def main():
    print("Starting test for wotan...")
    numpy.random.seed(seed=0)  # reproducibility

    # TESS test
    print('Loading TESS data from archive.stsci.edu...')
    filename = "https://archive.stsci.edu/hlsps/tess-data-alerts/" \
    "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"

    #filename = "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
    time, flux = load_file(filename)

    window_length = 0.5
    print("Detrending 1 (biweight)...")
    flatten_lc, trend_lc = flatten(
        time,
        flux,
        window_length,
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

    print("Detrending 2 (andrewsinewave)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='andrewsinewave', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.15471987987)

    print("Detrending 3 (welsch)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='welsch', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.16764691235)

    print("Detrending 4 (hodges)...")
    flatten_lc, trend_lc = flatten(time[:2000], flux[:2000], window_length, method='hodges', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 1947.0182174079318)

    print("Detrending 5 (median)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='median', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.122065014355)

    print("Detrending 6 (mean)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='mean', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.032473037714)

    print("Detrending 7 (trim_mean)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='trim_mean', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.095164910334)

    print("Detrending 8 (supersmoother)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='supersmoother', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18122.933205071175, decimal=5)

    print("Detrending 9 (huberspline)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='huberspline', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18123.07625225313, decimal=2)
    
    print("Detrending 10 (cofiam)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='cofiam', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18122.999983007176, decimal=2)
    
    """
    import matplotlib.pyplot as plt
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
