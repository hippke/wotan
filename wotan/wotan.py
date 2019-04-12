"""Wotan: is a free and open source algorithm to automagically remove stellar trends
from light curves for exoplanet transit detection.
"""
from __future__ import division
import numpy
from numpy import mean, median, array, abs, sort, inf, sin, exp, sum, pi, min, max, \
    full, append, concatenate, diff, where, add, float32, nan, isnan
from numba import jit


@jit(fastmath=True, nopython=True, cache=True)
def location_trim_mean(data, proportiontocut):
    """Return mean of array after trimming `proportiontocut` from both tails."""
    len_data = len(data)
    sorted_data = sort(data)
    cut_idx = int(len_data * proportiontocut)
    return mean(sorted_data[cut_idx:len_data-cut_idx])


@jit(fastmath=True, nopython=True, cache=True)
def location_hodges(data):
    """Hodges–Lehmann–Sen robust location estimator"""
    i = 0
    j = 0
    len_data = len(data)
    hodges = []    
    while i < len_data:
        while j < len_data:
            hodges.append((data[i] + data[j]) / 2)
            j += 1
        i += 1
        j = i
    return median(array(hodges))


@jit(fastmath=True, nopython=True, cache=True)
def location_iter(data, cval, ftol, method_code):
    """Robust location estimators"""

    # Numba can't handle string, so we're passing the location estimator as an int:
    # 1 : biweight
    # 2 : andrewsinewave
    # 3 : welsch
    # (the others are not relevant for location_iter)

    # Initial estimate for the central location
    delta_center = inf
    median_data = median(data)
    mad = median(abs(data - median_data))
    center = center_old = median_data

    # Neglecting this case was a bug in scikit-learn
    if mad == 0:
        return center

    # one expensive division here, instead of two per loop later
    blub = cval * mad
    cmad = 1 / (cval * mad)

    # Newton-Raphson iteration, where each result is taken as the initial value of the
    # next iteration. Stops when the difference of a round is below ``ftol`` threshold
    while abs(delta_center) > ftol:
        distance = data - center
        dmad = distance * cmad

        # Inlier weights
        # biweight
        if method_code == 1:  
            weight = (1 - dmad ** 2) ** 2
        # andrewsinewave
        if method_code == 2:
            # avoid division by zero  
            dmad[dmad==0] = 1e-10  
            weight = sin(dmad) / dmad
        # welsch
        if method_code == 3:  
            weight = exp(-(dmad**2) / 2)

        # Outliers with weight zero
        # biweight or welsch
        if method_code == 1 or method_code == 3:  
            weight[(abs(dmad) >= 1)] = 0
        # andrewsinewave
        else:  
            weight[(abs(dmad) >= pi)] = 0

        center += sum(distance * weight) / sum(weight)

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center
        center_old = center
    return center


@jit(fastmath=True, nopython=True, cache=True)
def running_segment(time, flux, window_length, edge_cutoff, cval, ftol, method_code):
    """Iterator for a single time-series segment"""

    # Numba can't handle string, so we're passing the location estimator as an int:
    # 1 : biweight
    # 2 : andrewsinewave
    # 3 : welsch
    # 4 : hodges
    # 5 : median
    # 6 : mean
    # 7 : trim_mean

    size = len(time)
    mean_all = full(size, nan)
    half_window = window_length / 2
    # 0 < Edge cutoff < half_window:
    if edge_cutoff > half_window:
        edge_cutoff = half_window

    # Pre-calculate border checks before entering the loop (reason: large speed gain)
    low_index = min(time) + edge_cutoff
    hi_index = max(time) - edge_cutoff
    idx_start = 0
    idx_end = 0

    for i in range(size-1):
        if time[i] > low_index and time[i] < hi_index:
            # Nice style would be:
            #   idx_start = numpy.argmax(time > time[i] - window_length/2)
            #   idx_end = numpy.argmax(time > time[i] + window_length/2)
            # But that's too slow (factor 10). Instead, we write:
            while time[idx_start] < time[i] - half_window:
                idx_start += 1
            while time[idx_end] < time[i] + half_window and idx_end < size-1:
                idx_end += 1

            # Get the location estimate for the segment in question
            # iterative method for: biweight, andrewsinewave, welsch            
            if method_code == 1 or method_code == 2 or method_code == 3:
                mean_all[i] = location_iter(
                    flux[idx_start:idx_end],
                    cval,
                    ftol,
                    method_code
                    )
            # hodges
            elif method_code == 4:
                mean_all[i] = location_hodges(flux[idx_start:idx_end])
            # median
            elif method_code == 5:
                mean_all[i] = median(flux[idx_start:idx_end])
            # mean
            elif method_code == 6:
                mean_all[i] = mean(flux[idx_start:idx_end])
            # trim_mean
            elif method_code == 7:
                mean_all[i] = location_trim_mean(
                    flux[idx_start:idx_end],
                    proportiontocut=cval)
            
    return mean_all


def get_gaps_indexes(time, break_tolerance):
    """Array indexes where ``time`` has gaps longer than ``break_tolerance``"""
    gaps = diff(time)
    gaps_indexes = where(gaps > break_tolerance)
    gaps_indexes = add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = concatenate(gaps_indexes).ravel()  # Flatten
    gaps_indexes = append(array([0]), gaps_indexes)  # Start
    gaps_indexes = append(gaps_indexes, array([len(time)+1]))  # End point
    return gaps_indexes


def flatten(time, flux, window_length, edge_cutoff=0, break_tolerance=None, cval=None,
            ftol=1e-6, return_trend=False, method='biweight'):
    """``flatten`` removes low frequency trends with a robust time-windowed slider.
    Parameters
    ----------
    time : array-like
        Time values
    flux : array-like
        Flux values for every time point
    window_length : float
        The length of the filter window in units of ``time`` (usually days).
        ``window_length`` must be a positive floating point value.
    method : string, default: `biweight`
        Determines detrending method and location estimator. A time-windowed slider is
        invoked for location estimators `median`, `biweight`, `hodges`, `welsch`, 
        `andrewsinewave`, `mean`, or `trim_mean`. Spline-based detrending is performed
        for `huberspline` and `untrendy`. A locally weighted scatterplot smoothing is
        performed for `lowess`
    break_tolerance : float, default: window_length/2
        If there are large gaps in time (larger than ``window_length``/2), flatten will
        split the flux into several sub-lightcurves and apply the filter to each
        individually. ``break_tolerance`` must be in the same unit as ``time`` (usually 
        days). To disable this feature, set ``break_tolerance`` to 0.
    edge_cutoff : float, default: None
        Trends near edges are less robust. Depending on the data, it may be beneficial
        to remove edges. The ``edge_cutoff`` defines the length (in units of time) to be 
        cut off each edge. Default: Zero. Cut off is maximally ``window_length``/2, as 
        this fills the window completely. Applies only to time-windowed sliders.
    cval : float
        Tuning parameter for the robust estimators. Default values are 5 (`biweight` and 
        `lowess`), 1.339 (`andrewsinewave`), 2.11 (`welsch`), 0.1 (`trim_mean`). A 
        ``cval`` of 6 for the biweight includes data up to 4 standard deviations from 
        the central location and has an efficiency of 98%. Another typical value for the
        biweight is 4.685 with 95% efficiency. Larger values for make the estimate more 
        efficient but less robust.
    ftol : float, default: 1e-6
        Desired precision of the final location estimate of the `biweight`, `welsch`,
        and `andrewsinewave`. All other methods use one-step estimates. The iterative 
        algorithm based on Newton-Raphson stops when the change in location becomes 
        smaller than ``ftol``. Default: `1e-6`, or 1ppm. Higher precision comes at 
        greater computational expense.
    return_trend : bool, default: False
        If `True`, the method will return a tuple of two elements
        (``flattened_flux``, ``trend_flux``) where ``trend_flux`` is the removed trend.
    Returns
    -------
    flatten_flux : array-like
        Flattened flux.
    trend_flux : array-like
        Trend in the flux. Only returned if ``return_trend`` is `True`.
    """
    # Numba can't handle string, so we're passing the location estimator as an int:
    if method == 'biweight':
        method_code = 1
    elif method == 'andrewsinewave':
        method_code = 2
    elif method == 'welsch':
        method_code = 3
    elif method == 'hodges':
        method_code = 4
    elif method == 'median':
        method_code = 5
    elif method == 'mean':
        method_code = 6
    elif method == 'trim_mean':
        method_code = 7

    # Default cval values for robust location estimators
    if cval is None:
        if method == 'biweight':
            cval = 5
        elif method == 'andrewsinewave':
            cval = 1.339
        elif method == 'welsch':
            cval = 2.11
        elif method == 'trim_mean':
            cval = 0.1  # 10 % on each side
        else: cval = 0  # avoid numba type inference error: None type multi with float

    # Maximum gap in time should be half a window size.
    # Any larger is nonsense,  because then the array has a full window of data
    if break_tolerance is None:
        break_tolerance = window_length / 2

    # Numba is very fast, but doesn't play nicely with NaN values
    # Therefore, we make new time-flux arrays with only the floating point values
    # All calculations are done within these arrays
    # Afterwards, the trend is transplanted into the original arrays (with the NaNs)
    time = array(time, dtype=float32)
    flux = array(flux, dtype=float32)
    mask = isnan(time * flux)
    time_compressed = numpy.ma.compressed(numpy.ma.masked_array(time, mask))
    flux_compressed = numpy.ma.compressed(numpy.ma.masked_array(flux, mask))

    # Get the indexes of the gaps
    gaps_indexes = get_gaps_indexes(time_compressed, break_tolerance=break_tolerance)
    trend_flux = array([])
    trend_segment = array([])

    # Iterate over all segments
    for i in range(len(gaps_indexes)-1):
        trend_segment = running_segment(
            time_compressed[gaps_indexes[i]:gaps_indexes[i+1]],
            flux_compressed[gaps_indexes[i]:gaps_indexes[i+1]],
            window_length,
            edge_cutoff,
            cval,
            ftol,
            method_code)
        trend_flux = append(trend_flux, trend_segment)

    # Insert results of non-NaNs into original data stream
    trend_lc = full(len(time), nan)
    mask = where(~mask)[0]
    for idx in range(len(mask)):
        trend_lc[mask[idx]] = trend_flux[idx]

    flatten_lc = flux / trend_lc
    if return_trend:
        return flatten_lc, trend_lc
    return flatten_lc
