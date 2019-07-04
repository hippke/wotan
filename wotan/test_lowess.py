from __future__ import print_function, division
from wotan import flatten, t14, slide_clip
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
    print("Starting tests for wotan...")
    import time as ttime

    numpy.random.seed(seed=1)
    points = 1000
    time = numpy.linspace(0, 30, points)
    flux = 1 + numpy.sin(time)  / points
    noise = numpy.random.normal(0, 0.0001, points)
    flux += noise

    for i in range(points):  
        if i % 75 == 0:
            flux[i:i+5] -= 0.0004  # Add some transits
            flux[i+50:i+52] += 0.0002  # and flares

    t1 = ttime.perf_counter()
    print("Detrending 22 (lowess)...")
    flatten_lc, trend_lc1 = flatten(
        time,
        flux,
        method='lowess',
        window_length=1,
        return_trend=True
        )
    t2 = ttime.perf_counter()
    print('time', t2 - t1)
    #numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18123.08085676265, decimal=2)


    print('lowess new')
    from lowess import lowess
    t3 = ttime.perf_counter()
    trend2 = lowess(time, flux, window_length=1)
    t4 = ttime.perf_counter()
    print(numpy.sum(trend2))
    print('time', t4 - t3)
    #assert numpy.sum(trend2) == 
    #1000.0255579207636
    #1000.025557920763
    #1000.0255579207638
    #1000.0255579207634
    #1000.0255155852465
    #1000.0255155852469
    #1000.025515585246
    #1000.025515585246

    import matplotlib.pyplot as plt
    plt.scatter(time, flux, s=1, color='black')
    plt.plot(time, trend_lc1, color='blue', linewidth=2)
    plt.plot(time, trend2, color='red', linewidth=2, linestyle='dashed')
    plt.show()
    plt.close()
    #plt.scatter(time, flatten_lc, s=1, color='black')
    #plt.show()

    path = 'https://archive.stsci.edu/hlsps/tess-data-alerts/'
    filename = "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
    time, flux = load_file(path + filename)
    from transitleastsquares import cleaned_array
    time, flux = cleaned_array(time, flux)
    trend2 = lowess(time, flux, window_length=1)
    plt.scatter(time, flux, s=1, color='black')
    plt.plot(time, trend2, color='red', linewidth=2, linestyle='dashed')
    plt.show()
    plt.close()

    print('All tests completed.')


if __name__ == '__main__':
    main()
