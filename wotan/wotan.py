"""Wotan: is a free and open source algorithm to automagically remove stellar trends
from light curves for exoplanet transit detection.
"""
from __future__ import division
import numpy
from numba import jit



@jit(fastmath=True, nopython=True, cache=True)
def biweight_location_iter(data, cval, ftol):
    """Robust location estimate using iterative Tukey's biweight"""

    # Initial estimate for the central location
    delta_center = numpy.inf
    median_data = numpy.median(data)
    mad = numpy.median(numpy.abs(data - median_data))
    center = center_old = median_data

    # Neglecting this case was a bug in scikit-learn
    if mad == 0:
        return center

    # one expensive division here, instead of two per loop later
    cmad = 1 / (cval * mad)

    # Newton-Raphson iteration, where each result is taken as the initial value of the
    # next iteration. Stops when the difference of a round is below ``ftol`` threshold
    while abs(delta_center) > ftol:
        distance = data - center

        # Inliers with Tukey's biweight
        weight = (1 - (distance * cmad) ** 2) ** 2

        # Outliers with weight zero
        weight[(numpy.abs(distance * cmad) >= 1)] = 0
        center += numpy.sum(distance * weight) / numpy.sum(weight)

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center
        center_old = center
    return center


@jit(fastmath=True, nopython=True, cache=True)
def running_segment(time, flux, window_length, edge_cutoff, cval, ftol):
    """Iterator for a single time-series segment"""

    size = len(time)
    mean_all = numpy.full(size, numpy.nan)
    half_window = window_length / 2
    # 0 < Edge cutoff < half_window:
    if edge_cutoff > half_window:
        edge_cutoff = half_window

    # Pre-calculate border checks before entering the loop (reason: large speed gain)
    low_index = numpy.min(time) + edge_cutoff
    hi_index = numpy.max(time) - edge_cutoff
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

            # Get the Tukey biweight mean for the segment in question
            mean_all[i] = biweight_location_iter(
                flux[idx_start:idx_end],
                cval,
                ftol
                )
    return mean_all


def get_gaps_indexes(time, break_tolerance):
    """Array indexes where ``time`` has gaps longer than ``break_tolerance``"""
    gaps = numpy.diff(time)
    gaps_indexes = numpy.where(gaps > break_tolerance)
    gaps_indexes = numpy.add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = numpy.concatenate(gaps_indexes).ravel()  # Flatten
    gaps_indexes = numpy.append(numpy.array([0]), gaps_indexes)  # Start
    gaps_indexes = numpy.append(gaps_indexes, numpy.array([len(time)+1]))  # End point
    return gaps_indexes


def flatten(time, flux, window_length, edge_cutoff=0, break_tolerance=None, cval=6,
            ftol=1e-6, return_trend=False):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """

    # Maximum gap in time should be half a window size.
    # Any larger is nonsense,  because then the array has a full window of data
    if break_tolerance is None:
        break_tolerance = window_length / 2

    # Numba is very fast, but doesn't play nicely with NaN values
    # Therefore, we make new time-flux arrays with only the floating point values
    # All calculations are done within these arrays
    # Afterwards, the trend is transplanted into the original arrays (with the NaNs)
    time = numpy.array(time, dtype=numpy.float32)
    flux = numpy.array(flux, dtype=numpy.float32)
    mask = numpy.isnan(time * flux)
    time_compressed = numpy.ma.compressed(numpy.ma.masked_array(time, mask))
    flux_compressed = numpy.ma.compressed(numpy.ma.masked_array(flux, mask))

    # Get the indexes of the gaps
    gaps_indexes = get_gaps_indexes(time_compressed, break_tolerance=break_tolerance)
    trend_flux = numpy.array([])
    trend_segment = numpy.array([])

    # Iterate over all segments
    for i in range(len(gaps_indexes)-1):
        trend_segment = running_segment(
            time_compressed[gaps_indexes[i]:gaps_indexes[i+1]],
            flux_compressed[gaps_indexes[i]:gaps_indexes[i+1]],
            window_length,
            edge_cutoff,
            cval,
            ftol)
        trend_flux = numpy.append(trend_flux, trend_segment)

    # Insert results of non-NaNs into original data stream
    trend_lc = numpy.full(len(time), numpy.nan)
    mask = numpy.where(~mask)[0]
    for idx in range(len(mask)):
        trend_lc[mask[idx]] = trend_flux[idx]

    flatten_lc = flux / trend_lc
    if return_trend:
        return flatten_lc, trend_lc
    return flatten_lc
