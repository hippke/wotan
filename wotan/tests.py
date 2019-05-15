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

    numpy.testing.assert_almost_equal(
        t14(R_s=1, M_s=1, P=365),
        0.6490025258902046)

    numpy.testing.assert_almost_equal(
        t14(R_s=1, M_s=1, P=365, small_planet=True),
        0.5403690143737738)
    print("Transit duration correct.")

    numpy.random.seed(seed=0)  # reproducibility

    print("Slide clipper...")
    points = 1000
    time = numpy.linspace(0, 30, points)
    flux = 1 + numpy.sin(time)  / points
    noise = numpy.random.normal(0, 0.0001, points)
    flux += noise

    for i in range(points):  
        if i % 75 == 0:
            flux[i:i+5] -= 0.0004  # Add some transits
            flux[i+50:i+52] += 0.0002  # and flares

    clipped = slide_clip(
    time,
    flux,
    window_length=0.5,
    low=3,
    high=2,
    method='mad',
    center='median'
    )
    numpy.testing.assert_almost_equal(numpy.nansum(clipped), 948.9926368754939)

    """
    import matplotlib.pyplot as plt
    plt.scatter(time, flux, s=3, color='black')
    plt.scatter(time, clipped, s=3, color='orange')
    plt.show()
    """

    # TESS test
    print('Loading TESS data from archive.stsci.edu...')
    path = 'https://archive.stsci.edu/hlsps/tess-data-alerts/'
    filename = "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
    #path = 'P:/P/Dok/tess_alarm/'
    #filename = "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
    #filename = 'P:/P/Dok/tess_alarm/hlsp_tess-data-alerts_tess_phot_00077031414-s02_tess_v1_lc.fits'
    #filename = 'tess2018206045859-s0001-0000000201248411-111-s_llc.fits'
    time, flux = load_file(path + filename)

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
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.15471987987, decimal=2)

    print("Detrending 3 (welsch)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='welsch', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.16764691235, decimal=2)

    print("Detrending 4 (hodges)...")
    flatten_lc, trend_lc = flatten(time[:1000], flux[:1000], window_length, method='hodges', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 994.0110525909206, decimal=2)

    print("Detrending 5 (median)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='median', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.122065014355, decimal=2)

    print("Detrending 6 (mean)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='mean', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.032473037714, decimal=2)

    print("Detrending 7 (trim_mean)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='trim_mean', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.095164910334, decimal=2)

    print("Detrending 8 (supersmoother)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='supersmoother', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18123.00632204841, decimal=2)

    print("Detrending 9 (hspline)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length, method='hspline', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18123.07625225313, decimal=2)

    print("Detrending 10 (cofiam)...")
    flatten_lc, trend_lc = flatten(time[:2000], flux[:2000], window_length, method='cofiam', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 1948.9999999987976, decimal=2)

    print("Detrending 11 (savgol)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length=301, method='savgol', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18123.003465539354, decimal=2)

    print("Detrending 12 (medfilt)...")
    flatten_lc, trend_lc = flatten(time, flux, window_length=301, method='medfilt', return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18123.22609806557, decimal=2)

    print("Detrending 12 (gp squared_exp)...")
    flatten_lc, trend_lc1 = flatten(
        time[:2000],
        flux[:2000],
        method='gp',
        kernel='squared_exp',
        kernel_size=10,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 1948.99958552324, decimal=2)

    print("Detrending 13 (gp squared_exp robust)...")
    flatten_lc, trend_lc1 = flatten(
        time[:2000],
        flux[:2000],
        method='gp',
        kernel='squared_exp',
        kernel_size=10,
        robust=True,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 1948.8820772313468, decimal=2)
    


    print("Detrending 14 (gp matern)...")
    flatten_lc, trend_lc2 = flatten(
        time[:2000],
        flux[:2000],
        method='gp',
        kernel='matern',
        kernel_size=10,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 1949.0001583058202, decimal=2)

    print("Detrending 15 (gp periodic)...")
    flatten_lc, trend_lc2 = flatten(
        time[:2000],
        flux[:2000],
        method='gp',
        kernel='periodic',
        kernel_size=1,
        kernel_period=10,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 1948.9999708985608, decimal=2)

    time_synth = numpy.linspace(0, 30, 200)
    flux_synth = numpy.sin(time_synth) + numpy.random.normal(0, 0.1, 200)
    flux_synth = 1 + flux_synth / 100
    time_synth *= 1.5
    print("Detrending 16 (gp periodic_auto)...")
    flatten_lc, trend_lc2 = flatten(
        time_synth,
        flux_synth,
        method='gp',
        kernel='periodic_auto',
        kernel_size=1,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 200, decimal=1)
    
    print("Detrending 17 (rspline)...")
    flatten_lc, trend_lc2 = flatten(
        time,
        flux,
        method='rspline',
        window_length=1,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18121.812790732245, decimal=2)

    print("Detrending 18 (huber)...")
    flatten_lc, trend_lc = flatten(
        time[:1000],
        flux[:1000],
        method='huber',
        window_length=0.5,
        edge_cutoff=0,
        break_tolerance=0.4,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 994.01102, decimal=2)
    
    print("Detrending 19 (winsorize)...")
    flatten_lc, trend_lc2 = flatten(
        time,
        flux,
        method='winsorize',
        window_length=0.5,
        edge_cutoff=0,
        break_tolerance=0.4,
        proportiontocut=0.1,
        return_trend=True)
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.064587196448, decimal=2)
    
    print("Detrending 20 (pspline)...")
    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method='pspline',
        return_trend=True
        )
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18121.832133916843, decimal=2)
    
    print("Detrending 21 (hampelfilt)...")
    flatten_lc, trend_lc5 = flatten(
        time,
        flux,
        method='hampelfilt',
        window_length=0.5,
        cval=3,
        return_trend=True
        )
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.158072498867, decimal=2)
    
    print("Detrending 22 (lowess)...")
    flatten_lc, trend_lc1 = flatten(
        time,
        flux,
        method='lowess',
        window_length=1,
        return_trend=True
        )
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18123.08085676265, decimal=2)

    print("Detrending 23 (huber_psi)...")
    flatten_lc, trend_lc1 = flatten(
        time,
        flux,
        method='huber_psi',
        window_length=0.5,
        return_trend=True
        )
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.122065014355, decimal=2)

    print("Detrending 24 (tau)...")
    flatten_lc, trend_lc2 = flatten(
        time,
        flux,
        method='tau',
        window_length=0.5,
        return_trend=True
        )
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 18119.02772621119, decimal=2)
    

    import numpy as np
    points = 1000
    time = np.linspace(0, 30, points)
    flux = 1 + np.sin(time)  / points
    noise = np.random.normal(0, 0.0001, points)
    flux += noise

    for i in range(points):  
        if i % 75 == 0:
            flux[i:i+5] -= 0.0004  # Add some transits
            flux[i+50:i+52] += 0.0002  # and flares


    print("Detrending 25a (hampel 17A)...")
    flatten_lc, trend_lc1 = flatten(
        time,
        flux,
        method='hampel',
        cval=(1.7, 3.4, 8.5),
        window_length=0.5,
        return_trend=True
        )

    print("Detrending 25b (hampel 25A)...")
    flatten_lc, trend_lc2 = flatten(
        time,
        flux,
        method='hampel',
        cval=(2.5, 4.5, 9.5),
        window_length=0.5,
        return_trend=True
        )
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 997.9994362858843, decimal=2)

    print("Detrending 26 (ramsay)...")
    flatten_lc, trend_lc3 = flatten(
        time,
        flux,
        method='ramsay',
        cval=0.3,
        window_length=0.5,
        return_trend=True
        )
    numpy.testing.assert_almost_equal(numpy.nansum(flatten_lc), 997.9974021484584, decimal=2)


    """
    import matplotlib.pyplot as plt
    plt.scatter(time, flux, s=1, color='black')
    plt.plot(time[:len(trend_lc1)], trend_lc1, color='blue', linewidth=2)
    plt.plot(time[:len(trend_lc1)], trend_lc2, color='red', linewidth=2, linestyle='dashed')
    plt.show()
    plt.close()
    #plt.scatter(time, flatten_lc, s=1, color='black')
    #plt.show()
    """


    print('All tests completed.')


if __name__ == '__main__':
    main()
