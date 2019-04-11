Python Interface
================

This describes the Python interface to Wotan.

TEST

.. autoclass:: DatabaseManager
    :members:

.. automodule:: wotan
    :members:

.. autofunction:: wotan.flatten

.. autofunction:: flatten

.. automodule:: wotan.flatten
    :members:


TEST

.. function:: flatten(time, flux, window_length, edge_cutoff=0, break_tolerance=None, cval=6, ftol=1e-6, return_trend=False)

Parameters
----------
time : array-like
    Time values
flux : array-like
    Flux values for every time point
window_length : float
    The length of the filter window in units of `t` (usually days).
    ``window_length`` must be a positive floating point value.
return_trend : bool (default: False)
    If `True`, the method will return a tuple of two elements
    (flattened_flux, trend_flux) where trend_flux is the removed trend.
break_tolerance : float
    If there are large gaps in time (larger than ``window_length/2``), flatten will 
    split the flux into several sub-lightcurves and apply the filter to each
    individually. `break_tolerance` must be in the same unit as ``t`` (usually days).
    To disable this feature, set `break_tolerance` to 0.
edge_cutoff : float
    Trends near edges are less robust. Depending on the data, it may be beneficial 
    to remove edges. The edge_cutoff length (in units of time) to be cut off each 
    edge. Default: Zero. Cut off is maximally `window_length/2`, as this fills the 
    window completely.
cval : float
    Tuning parameter for the Tukey biweight loss function. Default: cval=6 which 
    includes data up to 4 standard deviations from the central location and
    has an efficiency of 98%. Another typical values is c=4.685 with 95%
    efficiency. Larger values for cval make the estimate more efficient but less 
    robust.
ftol : float
    Desired precision of the final location estimate using Tukey's biweight.
    The iterative algorithms based on Newton-Raphson stops when the change in
    location becomes smaller than ``ftol``. Default: 1e-6, or 1ppm.
    Higher precision comes as greater computational expense.

Returns
-------
flatten_flux : array-like
    Flattened flux.
If ``return_trend`` is `True`, the method will also return:

trend_flux : array-like
    Trend in the flux

Example usage:

::

    intransit = transit_mask(t, period, duration, T0)
    print(intransit)
    >>> [False False False ...]
    plt.scatter(t[in_transit], y[in_transit], color='red')  # in-transit points in red
    plt.scatter(t[~in_transit], y[~in_transit], color='blue')  # other points in blue
