Usage examples
==============

As follows are usage example for all detrending methods offered by wotan. In all examples, the following synthetic data are used: 

::

    import numpy as np
    from wotan import flatten

    points = 1000
    time = np.linspace(0, 30, points)
    flux = 1 + ((np.sin(time) + time / 10 + time**1.5 / 100) / 1000)
    noise = np.random.normal(0, 0.0001, points)
    flux += noise
    for i in range(points):  
        if i % 75 == 0:
            flux[i:i+5] -= 0.0004  # Add some transits
            flux[i+50:i+52] += 0.0002  # and flares
    flux[300:400] = np.nan


Robust estimators with tuning constant
--------------------------------------

Some robust estimators can be tuned: ``biweight``, ``andrewsinewave``, ``welsch``, ``huber``, ``huber_psi``, ``hampel``, ``hampelfilt``, ``tau``. The ``hodges`` can not be tuned.


Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='biweight',
        window_length=0.5,    # The length of the filter window in units of ``time``
        edge_cutoff=0.5,      # length (in units of time) to be cut off each edge.
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        cval=5.0              # Tuning parameter for the robust estimators
        )

Which we can plot as follows:

::

    import matplotlib.pyplot as plt
    plt.scatter(time, flux, s=1, color='black')
    plt.plot(time, trend_lc, color='red', linewidth=2)
    plt.show()

    plt.close()
    plt.scatter(time, flatten_lc, s=1, color='black')
    plt.show()


.. note::

   Tuning constants ``cval`` are defined as multiples in units of median absolute deviation from the central location. Defaults are usually chosen to achieve high efficiency for Gaussian distributions. For example, for the ``biweight`` a ``cval`` of 6 includes data up to 4 standard deviations (6 median absolute deviations) from the central location and has an efficiency of 98%. Another typical value for the ``biweight`` is 4.685 with 95% efficiency. Larger values for make the estimate more efficient but less robust. The default for the ``biweight`` in wotan is 5, as it has shown the best results in the transit injection retrieval experiment. The other defaults are from the literature.

   - ``biweight`` 5
   - ``andrewsinewave`` 1.339
   - ``welsch`` 2.11
   - ``huber`` 1.5
   - ``huber_psi`` 1.28
   - ``hampel`` (1.7, 3.4, 8.5)
   - ``hampelfilt`` 3
   - ``ramsay`` 0.3
   - ``tau``: 4.5

   The ``hampel`` has a 3-part descending function, known also as (a,b,c). Its tuning constant ``cval`` must be given as a tuple of 3 values. Typical values are (1.7, 3.4, 8.5) called "17A"; and (2.5, 4.5, 9.5) called "25A". With values given as multiples of the median absolute deviation, the 25A can be stated equivalently: a' = 1.686, b' = 3.035 c' = 6.408 as multiples of the standard deviation.


Trimmed methods
---------------

There are 3 methods which first focus on outlier treatment, followed by taking the mean in a second stage: ``trim_mean``, ``winsorize`` and ``hampelfilt``. 

- The ``hampelfilt`` was already discussed in the previous section because its threshold is defined as ``cval`` times the median absolute deviation, beyond which it replaces values with the median.
- The ``trim_mean`` deletes the fraction ``proportiontocut`` from both sides of the distribution.
- The ``winsorize`` replaces the fraction ``proportiontocut`` from both sides of the distribution with the remaining values at the edges.

Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='trim_mean',
        window_length=0.5,    # The length of the filter window in units of ``time``
        edge_cutoff=0.5,      # length (in units of time) to be cut off each edge.
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        proportiontocut=0.1   # Cut 10% off both ends
        )


median, mean
------------

These methods ignore the parameters ``proportiontocut`` and ``cval``


Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='median',
        window_length=0.5,    # The length of the filter window in units of ``time``
        edge_cutoff=0.5,      # length (in units of time) to be cut off each edge.
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


medfilt
-------

This method is cadence-based. Included to compare to the time-windowed ``median``. The parameter ``window_length`` is now in units of cadence (i.e., array data points). It ignores the parameters ``edge_cutoff`` and ``break_tolerance``.


Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='medfilt',
        window_length=31 ,    # The length of the filter window in cadences
        return_trend=True,    # Return trend and flattened light curve
        )


Spline: robust rspline
----------------------

Spline with iterative sigma-clipping. It does not provide ``edge_cutoff``, but benefits greatly from using a sensible ``break_tolerance``. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='rspline',
        window_length=0.5,    # The knot distance in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


Spline: robust hspline
----------------------

Spline with robust Huber-estimator (linear and quadratic loss). It does not provide ``edge_cutoff``, but benefits greatly from using a sensible ``break_tolerance``. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='hspline',
        window_length=0.5,    # The knot distance in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


Spline: robust penalized pspline
--------------------------------

Spline with iterative sigma-clipping. Auto-determination of ``window_length``. It does not provide ``edge_cutoff``, but benefits greatly from using a sensible ``break_tolerance``. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='pspline',
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


Lowess / Loess
--------------

Locally weighted scatterplot smoothing (Cleveland 1979). Offers segmentation (``break_tolerance``), but no edge clipping (``edge_cutoff``). For similar results compared to other spline-based methods or sliders, use a ``window_length`` about twice as long. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='lowess',
        window_length=1,      # The length of the filter window in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


CoFiAM
--------------

Cosine Filtering with Autocorrelation Minimization. Does not provide ``edge_cutoff``, but benefits greatly from using a sensible ``break_tolerance``. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='cofiam',
        window_length=0.5,    # Protected window span in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


Fitting of sines and cosines
----------------------------

Fits a sum of sines and cosines, where the highest order is determined by the protected window span ``window_length`` in units of ``time``. A robustification (iterative sigma-clipping of 2-sigma outliers until convergence) is available by setting the parameter ``robust=True``. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='cosine',
        robust='True',        # iterative sigma-clipping of 2-sigma outliers until convergence
        window_length=0.5,    # Protected window span in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


SuperSmoother
--------------

Friedman's (1984) Super-Smoother, a local linear regression with adaptive bandwidth. Does not provide ``edge_cutoff``, but benefits greatly from using a sensible ``break_tolerance``. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='supersmoother',
        window_length=0.5,    # The knot distance in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        cval=None             # Bass enhancement (smoothness)
        )

.. note::

   ``cval`` determines the bass enhancement (smoothness) and can be `None` or in the range 0 < ``cval`` < 10. Smaller values make the trend more flexible to fit out small variations.


Savitzky-Golay savgol
---------------------

Sliding segments are fit with polynomials (Savitzky & Golay 1964). This filter is cadence-based (not time-windowed), so that ``window_length`` must be an integer value. If an even integer is provided, it is made uneven (a requirement) by adding 1. The polyorder is set by ``cval`` (default: 2 - the best value from our experiments). Does not provide ``edge_cutoff``, but benefits from using a sensible ``break_tolerance``. 

Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='savgol',
        cval=2,               # Defines polyorder
        window_length=51,     # The window length in cadences
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


Gaussian Processes
------------------

Available kernels are :

- ``squared_exp`` Squared-exponential kernel, with option for iterative sigma-clipping
- ``matern`` Matern 3/2 kernel, with option for iterative sigma-clipping
- ``periodic`` Periodic kernel informed by a user-specified period
- ``periodic_auto`` Periodic kernel informed by a Lomb-Scargle periodogram pre-search

GPs do not provide ``edge_cutoff``, but benefit from using a sensible ``break_tolerance``. 

Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='gp',
        kernel='squared_exp', # GP kernel choice
        kernel_size=10,       # GP kernel length
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )

.. note::

   The sensible ``kernel_size`` varies between kernels.


A robustification (iterative sigma-clipping of 2-sigma outliers until convergence) is available by setting the parameter ``robust=True``:

::

    flatten_lc, trend_lc = flatten(
        time,                 # Array of time values
        flux,                 # Array of flux values
        method='gp',          
        kernel='squared_exp', # GP kernel choice
        kernel_size=10,       # GP kernel length
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        robust=True,          # Robustification using iterative sigma clipping
        return_trend=True,    # Return trend and flattened light curve
        )

Here we can simply swap ``kernel='squared_exp'`` for ``kernel='matern'`` and play with ``kernel_size`` to get a very similar result.

In the presence of strong periodicity, we can also use the periodic kernel. This version does not support robustification. If we know the period, we can do this.

::

    flatten_lc2, trend_lc2 = flatten(
        time,                  # Array of time values
        flux,                  # Array of flux values
        method='gp',
        kernel='periodic',     # GP kernel choice
        kernel_period=2*3.14,  # GP kernel period
        kernel_size=10,        # GP kernel length
        break_tolerance=0.5,   # Split into segments at breaks longer than that
        return_trend=True,     # Return trend and flattened light curve
        )

Usually, however, it is better to let wotan detect the period. We can do this by setting ``kernel='periodic_auto'``. Then, a Lomb-Scargle periodogram is calculated, and the strongest peak is used as the period. In addition, a Matern kernel is added to consume the remaining non-periodic variation. This version does not support robustification. Example:

::

    flatten_lc2, trend_lc2 = flatten(
        time,                    # Array of time values
        flux,                    # Array of flux values
        method='gp',
        kernel='periodic_auto',  # GP kernel choice
        kernel_size=10,          # GP kernel length
        break_tolerance=0.5,     # Split into segments at breaks longer than that
        return_trend=True,       # Return trend and flattened light curve
        )
