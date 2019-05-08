from __future__ import print_function, division
from numba import jit
import numpy
from numpy import mean, median, full, nan
from wotan.location_estimates import (
    location_iter, hodges, trim_mean, winsorize_mean, hampelfilt, huber_psi, tau
    )
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
    # 8 : winsorize
    # 9 : hampelfilt
    # 10: huber_one
    # 11: tau

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
            f = flux[idx_start:idx_end]
            if method_code == 1 or method_code == 2 or method_code == 3:
                mean_all[i] = location_iter(f, cval, method_code)
            # hodges
            elif method_code == 4:
                mean_all[i] = hodges(f)
            # median
            elif method_code == 5:
                mean_all[i] = median(f)
            # mean
            elif method_code == 6:
                mean_all[i] = mean(f)
            # trim_mean
            elif method_code == 7:
                mean_all[i] = trim_mean(f, proportiontocut=cval)
            # winsorize
            elif method_code == 8:
                mean_all[i] = winsorize_mean(f, proportiontocut=cval)
            # hampel
            elif method_code == 9:
                mean_all[i] = hampelfilt(f, cval=cval)
            # huber_psi
            elif method_code == 10:
                mean_all[i] = huber_psi(f, cval=cval)
            # tau
            elif method_code == 11:
                mean_all[i] = tau(f, cval=cval)
    return mean_all


def running_segment_slow(time, flux, window_length, edge_cutoff, cval, method):
    """With the import, we can't use numba. Thus this here is separate and slower"""
    try:
        # import statsmodels.api as sm
        from statsmodels.api import robust, RLM
    except:
        raise ImportError('Could not import statsmodels')
    size = len(time)
    mean_all = full(size, nan)
    half_window = window_length / 2
    # 0 < Edge cutoff < half_window:
    if edge_cutoff > half_window:
        edge_cutoff = half_window
    # Flat regression for ``hampel```and ``ramsay``
    flat = numpy.ones(len(flux))
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
            # Huber is not numerically stable
            # If it does not converge, we fall back to the median
            if method == 'huber':
                try:
                    huber = robust.scale.Huber(
                        maxiter=constants.MAXITER_HUBER,
                        tol=constants.FTOL,
                        c=cval
                        )
                    mean_all[i], error = huber(flux[idx_start:idx_end])
                except:
                    mean_all[i] = median(flux[idx_start:idx_end])
            elif method == 'hampel':
                points = len(flux[idx_start:idx_end])
                mean_all[i] = RLM(
                    flux[idx_start:idx_end],
                    flat[:points],
                    M=robust.norms.Hampel(
                        a=cval[0],
                        b=cval[1],
                        c=cval[2])
                    ).fit().params[0]
            elif method == 'ramsay': 
                points = len(flux[idx_start:idx_end])
                mean_all[i] = RLM(
                    flux[idx_start:idx_end],
                    flat[:points],
                    M=robust.norms.RamsayE(
                        a=cval)
                    ).fit().params[0]
    return mean_all
