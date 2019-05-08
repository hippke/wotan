Wotan Interface
================


Detrending with the ``flatten`` module 
--------------------------------------

.. automodule:: flatten.flatten

Usage example:

::

    import numpy as np
    from astropy.io import fits

    def load_file(filename):
        """Loads a TESS *spoc* FITS file and returns TIME, PDCSAP_FLUX"""
        hdu = fits.open(filename)
        time = hdu[1].data['TIME']
        flux = hdu[1].data['PDCSAP_FLUX']
        flux[flux == 0] = np.nan
        return time, flux

    print('Loading TESS data from archive.stsci.edu...')
    path = 'https://archive.stsci.edu/hlsps/tess-data-alerts/'
    filename = "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
    time, flux = load_file(path + filename)

    # Use wotan to detrend
    from wotan import flatten
    flatten_lc1, trend_lc1 = flatten(time, flux, window_length=0.75, return_trend=True, method='mean')
    flatten_lc2, trend_lc2 = flatten(time, flux, window_length=0.75, return_trend=True, method='biweight')

    # Plot the result
    import matplotlib.pyplot as plt
    plt.scatter(time, flux, s=1, color='black')
    plt.plot(time, trend_lc1, linewidth=2, color='red')
    plt.plot(time, trend_lc2, linewidth=2, color='blue')
    plt.xlim(min(time), 1365)
    plt.show()


Choosing the right window size
--------------------------------------

Shorter windows (or knot distances, smaller kernels...) remove stellar variability more effectively, but suffer a larger risk of removing the desired signal (the transit) as well. What is the right window size?

For the time-windowed sliders, the window should be 2-3 times longer than the transit duration (for details, read [the paper](www). The transit duration is

:math:`T_{14,{\rm max}} = (R_{\rm s}+R_{\rm p}) \left( \frac{4P}{\pi G M_{\rm s}} \right)^{1/3}`

for a central transit on a circular orbit. If you have a prior on the stellar mass and radius, and a (perhaps maximum) planetary period, ``wotan`` offers a convenience function to calculate :math:`T_{14,{\rm max}}`:

.. automodule:: t14.t14

As an example, we can calculate the duration of an Earth-Sun transit:

::

    from wotan import t14
    tdur = t14(R_s=1, M_s=1, P=365, small_planet=True)
    print(tdur)

This should print ~0.54 (days), or about 13 hours. To protect a transit that long, it is reasonable to choose a window size of 3x as long, or about 1.62 days. With the ``biweight`` time-windowed slider, we would detrend with these settings:

::

    from wotan import t14, flatten
    tdur = t14(R_s=1, M_s=1, P=365, small_planet=True)
    flatten_lc = flatten(time, flux, window_length=3 * tdur)



Removing outliers after detrending
--------------------------------------

With robust detrending methods, the trend line (and thus the detrended data) may be unaffected by outliers. In the actual data, however, outliers are still present after detrending. For many purposes, it is acceptable to clip this:

::

    from wotan import t14, flatten

from astropy.stats import sigma_clip
flux = sigma_clip(flux, sigma_upper=3, sigma_lower=20)
