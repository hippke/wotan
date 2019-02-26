Python Interface
================

This describes the Python interface to Wotan.


Define data for a search
------------------------

.. function:: detrend(t, y, dy)

:t: *(array)* Time series of the data (**in units of days**)
:y: *(array)* Flux series of the data, so that ``1`` is nominal flux (out of transit) and ``0`` is darkness. A transit may be represented by a flux of e.g., ``0.99``
:dy: *(array, optional)* Measurement errors of the data

:max_window: *(float, default=0.75)* Maximum size of the moving window (in units of time)
:min_window: *(float, default=0.25)* Minimum size of the moving window (in units of time)
:threshold: *(float, default=0.05)* Allowed maximum power in a Lomb-Scargle periodogram. Will be used to reduce the window until either ``min_window`` or ``threshold`` are reached
:min_fraction: *(float, default=None)* Minimum fractional number of observations in window required to have a value (otherwise result is ``NaN``)
:estimator: *(str, default='biweight')* Choice of robust estimator. Choices: ``biweight`` (Tukey Bisquared Weight), ``huber``, ``theil_sen``
:sigma_clip_upper: *(float, default=3)* Threshold for values to be clipped. Values which are this number of standard deviations above the mean (within the current window) will be replaced with ``NaN``
:sigma_clip_lower: *(float, default=20)* Threshold for values to be clipped. Values which are this number of standard deviations below the mean (within the current window) will be replaced with ``NaN``


Returns

:trend: *(numpy array)* XYZ


Example usage:

::

    intransit = transit_mask(t, period, duration, T0)
    print(intransit)
    >>> [False False False ...]
    plt.scatter(t[in_transit], y[in_transit], color='red')  # in-transit points in red
    plt.scatter(t[~in_transit], y[~in_transit], color='blue')  # other points in blue
