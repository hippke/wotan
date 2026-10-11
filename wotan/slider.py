from __future__ import print_function, division
from numba import jit, prange
import numpy as np
from numpy import mean, median, full, nan
from wotan.location_estimates import (
    location_iter, location_iter_core, hodges, trim_mean, winsorize_mean, hampelfilt, huber_psi, tau
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


@jit(fastmath=True, nopython=True, cache=True)
def _insert(buf, n, value):
    """Insert ``value`` into the sorted ``buf[:n]``; returns the new length"""
    pos = np.searchsorted(buf[:n], value)
    for k in range(n, pos, -1):
        buf[k] = buf[k - 1]
    buf[pos] = value
    return n + 1


@jit(fastmath=True, nopython=True, cache=True)
def _remove(buf, n, value):
    """Remove one occurrence of ``value`` from the sorted ``buf[:n]``; returns new length"""
    pos = np.searchsorted(buf[:n], value)
    for k in range(pos, n - 1):
        buf[k] = buf[k + 1]
    return n - 1


@jit(fastmath=True, nopython=True, cache=True)
def _median_mad_sorted(s):
    """Median and median absolute deviation of the sorted array ``s`` in O(n)"""
    n = len(s)
    if n % 2 == 1:
        med = s[n // 2]
    else:
        med = (s[n // 2 - 1] + s[n // 2]) / 2
    # |s - med| are two ascending runs (left of med, reversed, and right of med).
    # Merge them up to the middle element(s).
    r = np.searchsorted(s, med)
    l = r - 1
    k1 = (n - 1) // 2
    k2 = n // 2
    v1 = 0.0
    v2 = 0.0
    for k in range(k2 + 1):
        if l < 0:
            d = s[r] - med
            r += 1
        elif r >= n:
            d = med - s[l]
            l -= 1
        else:
            dl = med - s[l]
            dr = s[r] - med
            if dl <= dr:
                d = dl
                l -= 1
            else:
                d = dr
                r += 1
        if k == k1:
            v1 = d
        if k == k2:
            v2 = d
    return med, (v1 + v2) / 2


@jit(fastmath=True, nopython=True, cache=True)
def _sliding_chunk(time, flux, mask, idx_start, idx_end, i0, i1, low_index, hi_index,
                   cval, method_code, buf, out):
    """Points i0..i1-1 with a sorted window buffer that is updated as the window slides
    (instead of copying and partially sorting every window twice for median and MAD)"""
    n = 0
    cur_lo = -1
    cur_hi = -1
    for i in range(i0, i1):
        if not (time[i] >= low_index and time[i] <= hi_index):
            continue
        lo = idx_start[i]
        hi = idx_end[i]
        if cur_lo < 0 or lo >= cur_hi:  # (re)build the buffer from scratch
            n = 0
            for j in range(lo, hi):
                if mask[j] > 0:
                    buf[n] = flux[j]
                    n += 1
            buf[:n].sort()
        else:  # windows only move forward: drop leaving, add entering values
            for j in range(cur_lo, lo):
                if mask[j] > 0:
                    n = _remove(buf, n, flux[j])
            for j in range(cur_hi, hi):
                if mask[j] > 0:
                    n = _insert(buf, n, flux[j])
        cur_lo = lo
        cur_hi = hi
        if n == 0:
            continue  # stays NaN
        med, mad = _median_mad_sorted(buf[:n])
        if method_code == 5:
            out[i] = med
        elif mad == 0:
            out[i] = med
        else:
            out[i] = location_iter_core(buf[:n], med, mad, cval, method_code)


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

    # biweight, andrewsinewave, welsch, median: sliding sorted window, in parallel chunks
    if method_code == 1 or method_code == 2 or method_code == 3 or method_code == 5:
        max_window = 1
        for i in range(size):
            max_window = max(max_window, idx_end[i] - idx_start[i])
        # Independent chunks of >= 4 windows (each chunk starts with one full sort);
        # the count does not depend on the thread count, so results are reproducible
        n_chunks = max(1, min(256, size // (4 * max_window)))
        for c in prange(n_chunks):
            i0 = c * size // n_chunks
            i1 = (c + 1) * size // n_chunks
            buf = np.empty(max_window)
            _sliding_chunk(time, flux, mask, idx_start, idx_end, i0, i1, low_index,
                           hi_index, cval, method_code, buf, mean_all)
        return mean_all

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
