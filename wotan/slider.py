from __future__ import print_function, division
from numba import jit, prange
import numpy as np
from numpy import mean, median, full, nan
from wotan.location_estimates import (
    location_iter, hodges, trim_mean, winsorize_mean, hampelfilt, huber_psi, tau
    )
import wotan.constants as constants


@jit(fastmath=True, nopython=True, cache=True)
def _location(f, cval, method_code):
    """Location estimate of the (unmasked) window values ``f``"""
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
    if len(f) == 0:
        return np.nan
    # iterative method for: biweight, andrewsinewave, welsch
    if method_code == 1 or method_code == 2 or method_code == 3:
        return location_iter(f, cval, method_code)
    elif method_code == 4:
        return hodges(f)
    elif method_code == 5:
        return median(f)
    elif method_code == 6:
        return mean(f)
    elif method_code == 7:
        return trim_mean(f, proportiontocut=cval)
    elif method_code == 8:
        return winsorize_mean(f, proportiontocut=cval)
    elif method_code == 9:
        return hampelfilt(f, cval=cval)
    elif method_code == 10:
        return huber_psi(f, cval=cval)
    elif method_code == 11:
        return tau(f, cval=cval)
    return np.nan


@jit(fastmath=True, nopython=True, cache=True, parallel=True)
def running_segment(time, flux, mask, window_length, edge_cutoff, cval, method_code):
    """Iterator for a single time-series segment using time-series window sliders.
    Each point is independent, so the points are processed in parallel threads."""
    size = len(time)
    mean_all = full(size, nan)
    half_window = window_length / 2
    # 0 < Edge cutoff < half_window:
    if edge_cutoff > half_window:
        edge_cutoff = half_window

    # Pre-calculate border checks before entering the loop (reason: large speed gain)
    low_index = np.min(time) + edge_cutoff
    hi_index = np.max(time) - edge_cutoff

    # Window of point i: time[i] - half_window <= time < time[i] + half_window
    # (``time`` is sorted, so these are the first indexes at/above the window edges)
    idx_start = np.searchsorted(time, time - half_window)
    idx_end = np.searchsorted(time, time + half_window)

    # ``mask`` holds 1 for usable and 0 for masked samples. It is applied by selecting
    # (not by manipulating the flux), so any flux value, including <= 0, is valid.
    for i in prange(size):
        if time[i] >= low_index and time[i] <= hi_index:
            lo = idx_start[i]
            hi = idx_end[i]
            f = flux[lo:hi][mask[lo:hi] > 0]
            mean_all[i] = _location(f, cval, method_code)
    return mean_all


def running_segment_slow(time, flux, mask, window_length, edge_cutoff, cval, method):
    """With the import, we can't use numba. Thus this here is separate and slower"""
    try:
        from statsmodels.api import robust, RLM
    except ImportError:
        raise ImportError('Could not import statsmodels')
    size = len(time)
    mean_all = full(size, nan)
    half_window = window_length / 2
    # 0 < Edge cutoff < half_window:
    if edge_cutoff > half_window:
        edge_cutoff = half_window

    # Flat regression for ``hampel```and ``ramsay``
    flat = np.ones(len(flux))

    # Pre-calculate border checks before entering the loop (reason: large speed gain)
    low_index = np.min(time) + edge_cutoff
    hi_index = np.max(time) - edge_cutoff
    idx_start = 0
    idx_end = 0

    # Set masked values to nan (on a copy). Filter out NaNs later.
    masked_flux = np.where(mask == 0, np.nan, flux)

    for i in range(size):
        if time[i] >= low_index and time[i] <= hi_index:
            # Nice style would be:
            #   idx_start = np.argmax(time > time[i] - window_length/2)
            #   idx_end = np.argmax(time > time[i] + window_length/2)
            # But that's too slow (factor 10). Instead, we write:
            while time[idx_start] < time[i] - half_window:
                idx_start += 1
            while idx_end < size and time[idx_end] < time[i] + half_window:
                idx_end += 1

            # drop the masked values
            f = masked_flux[idx_start:idx_end]
            f = f[~np.isnan(f)]
            if len(f) <= 1:
                mean_all[i] = np.nan
            else:
                if method == 'huber':
                    try:
                        huber = robust.scale.Huber(
                            maxiter=constants.MAXITER_HUBER,
                            tol=constants.FTOL,
                            c=cval
                            )
                        mean_all[i], error = huber(f)
                    # Huber is not numerically stable
                    # If it does not converge, we fall back to the median
                    except Exception:
                        mean_all[i] = median(f)
                elif method == 'hampel':
                    mean_all[i] = RLM(
                        f,
                        flat[:len(f)],
                        M=robust.norms.Hampel(
                            a=cval[0],
                            b=cval[1],
                            c=cval[2])
                        ).fit().params[0]
                elif method == 'ramsay': 
                    mean_all[i] = RLM(
                        f,
                        flat[:len(f)],
                        M=robust.norms.RamsayE(
                            a=cval)
                        ).fit().params[0]
    return mean_all
