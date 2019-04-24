from __future__ import print_function, division
from numba import jit
import numpy
from numpy import mean, median, full, nan
from wotan.location_estimates import location_iter, hodges, trim_mean, winsorize_mean
import wotan.constants as constants


@jit(fastmath=True, nopython=True, cache=True)
def running_segment(time, flux, window_length, edge_cutoff, cval, method_code):
    """Iterator for a single time-series segment using time-series window sliders"""

    # Numba can't handle strings, so we're passing the location estimator as an int:
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

            # Get the location estimate for the segment in question
            # iterative method for: biweight, andrewsinewave, welsch
            if method_code == 1 or method_code == 2 or method_code == 3:
                mean_all[i] = location_iter(
                    flux[idx_start:idx_end],
                    cval,
                    method_code
                    )
            # hodges
            elif method_code == 4:
                mean_all[i] = hodges(flux[idx_start:idx_end])
            # median
            elif method_code == 5:
                mean_all[i] = median(flux[idx_start:idx_end])
            # mean
            elif method_code == 6:
                mean_all[i] = mean(flux[idx_start:idx_end])
            # trim_mean
            elif method_code == 7:
                mean_all[i] = trim_mean(
                    flux[idx_start:idx_end],
                    proportiontocut=cval)
            # winsorize
            elif method_code == 8:
                mean_all[i] = winsorize_mean(
                    flux[idx_start:idx_end],
                    proportiontocut=cval)
    return mean_all


def running_segment_huber(time, flux, window_length, edge_cutoff, cval):

    # DUPLICATE, REFACTOR. Due to statsmodel import required only for huber
    # With the import, we can't use numba. thus this here is slower

    size = len(time)
    mean_all = full(size, nan)
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
            try:
                import statsmodels.api as sm
            except:
                raise ImportError('Could not import statsmodels')
            huber = sm.robust.scale.Huber(
                maxiter=constants.MAXITER_HUBER,
                tol=constants.FTOL,
                c=cval
                )
            mean_all[i], error = huber(flux[idx_start:idx_end])
    return mean_all
