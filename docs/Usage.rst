Usage examples
==============

As follows are usage example for all detrending methods offered by wotan.

In all examples, the following synthetic data are used: 

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


Biweight, andrewsinewave, welsch, huber, hampel
-----------------------------------------------

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

.. note::

   For the ``biweight``, a ``cval`` of 6 includes data up to 4 standard deviations (6 median absolute deviations) from the central location and has an efficiency of 98%. Another typical value for the ``biweight`` is 4.685 with 95% efficiency. Larger values for make the estimate more efficient but less robust.
   The default for the ``biweight`` in wotan is 5, as it has shown the best results in the transit injection retrieval experiment


.. note::
   For the other estimators, the same logic applies. However, their default values are different, following the standard statistics literature. It is typically set to achieve ~95% efficiency for Gaussian data.

   - ``andrewsinewave`` 1.339
   - ``welsch`` 2.11
   - ``huber`` 1.5
   - ``hampel`` 3


trim_mean, winsorize
--------------------

There are 3 methods which first focus on outlier treatment, followed by taking the mean in a second stage: ``trim_mean``, ``winsorize`` and ``hampel``. The ``hampel`` is already listed above because its threshold is defined as ``cval`` times the median absolute deviation, beyond which it replaced values with the median.

The ``trim_mean`` deletes the fraction ``proportiontocut`` from both sides of the distribution.

The ``winsorize`` replaces the fraction ``proportiontocut`` from both sides of the distribution with the remaining values at the edges.

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
        time,
        flux,
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
        time,
        flux,
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
        time,
        flux,
        method='pspline',
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


Lowess / Loess
--------------

Locally weighted scatterplot smoothing (Cleveland 1979). Offers segmentation (``break_tolerance``), but no edge clipping (``edge_cutoff``). For similar results compared to other spline-based methods or sliders, use a ``window_length`` about twice as long. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method='lowess',
        window_length=1,      # The length of the filter window in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


CoFiAM
--------------

Cosine Filtering with Autocorrelation Minimization. Does not provide ``edge_cutoff``, but benefits greatly from using a sensible ``break_tolerance``. Example usage:

    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method='cofiam',
        window_length=0.5,    # The knot distance in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


SuperSmoother
--------------

Friedman's (1984) Super-Smoother, a local linear regression with adaptive bandwidth. Does not provide ``edge_cutoff``, but benefits greatly from using a sensible ``break_tolerance``. Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method='supersmoother',
        window_length=0.5,    # The knot distance in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        cval=None             # 
        )

.. note::

   ``cval`` determines the bass enhancement (smoothness) and can be `None` or in the range 0 < ``cval`` < 10. Smaller values make the trend more flexible to fit out small variations.


Savitzky-Golay savgol
---------------------

Sliding segments are fit with polynomials (Savitzky & Golay 1964). This filter is cadence-based (not time-windowed), so that ``window_length`` must be an integer value. If an even integer is provided, it is made uneven (a requirement) by adding 1. The polyorder is hard-coded to 2 - the best value from our experiments. Does not provide ``edge_cutoff``, but benefits from using a sensible ``break_tolerance``. 

Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method='savgol',
        window_length=51,    # The knot distance in units of ``time``
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )


Gaussian Processes
------------------

Available kernels are 

- ``squared_exp`` Squared-exponential kernel, with option for iterative sigma-clipping
- ``matern`` Matern 3/2 kernel, with option for iterative sigma-clipping
- ``periodic`` Periodic kernel informed by a user-specified period
- ``periodic_auto`` Periodic kernel informed by a Lomb-Scargle periodogram pre-search

GPs do not provide ``edge_cutoff``, but benefit from using a sensible ``break_tolerance``. 

Example usage:

::

    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method='gp',
        kernel='squared_exp',
        kernel_size=10,
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        return_trend=True,    # Return trend and flattened light curve
        )

.. note::

   The sensible ``kernel_size`` varies between kernels.


A robustification (iterative sigma-clipping of 2-sigma outliers until convergence) is available by setting the parameter ``robust=True``:

::

    flatten_lc, trend_lc = flatten(
        time,
        flux,
        method='gp',
        kernel='squared_exp',
        kernel_size=10,
        break_tolerance=0.5,  # Split into segments at breaks longer than that
        robust=True,
        return_trend=True,    # Return trend and flattened light curve
        )

Here we can simply swap ``kernel='squared_exp'`` for ``kernel='matern'`` and play with ``kernel_size`` to get a very similar result.

In the presence of strong periodicity, we can also use the periodic kernel. This version does not support robustification. If we know the period, we can do this.

::

flatten_lc2, trend_lc2 = flatten(
    time,
    flux,
    method='gp',
    kernel='periodic',
    kernel_period=2*3.141592,
    kernel_size=10,
    break_tolerance=0.5,
    return_trend=True,
    )

Usually, however, it is better to let wotan detect the period. We can do this by setting ``kernel='periodic_auto'``. Then, a Lomb-Scargle periodogram is calculated, and the strongest peak is used as the period. In addition, a Matern kernel is added to consume the remaining non-periodic variation. This version does not support robustification. Example:

::

flatten_lc2, trend_lc2 = flatten(
    time,
    flux,
    method='gp',
    kernel='periodic_auto',
    kernel_size=10,
    break_tolerance=0.5,
    return_trend=True,
    )
