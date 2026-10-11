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
def _window_bounds(time, half_window):
    """First indexes with time >= time[i] - half_window and >= time[i] + half_window,
    as np.searchsorted(time, time -/+ half_window), by one linear sweep (the queries
    are sorted as well)"""
    size = len(time)
    idx_start = np.empty(size, dtype=np.int64)
    idx_end = np.empty(size, dtype=np.int64)
    lo = 0
    hi = 0
    for i in range(size):
        t_lo = time[i] - half_window
        t_hi = time[i] + half_window
        while lo < size and time[lo] < t_lo:
            lo += 1
        while hi < size and time[hi] < t_hi:
            hi += 1
        idx_start[i] = lo
        idx_end[i] = hi
    return idx_start, idx_end


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
def _replace(buf, n, old, new):
    """Replace one occurrence of ``old`` by ``new`` in the sorted ``buf[:n]``, moving
    only the elements between the two positions"""
    k = np.searchsorted(buf[:n], old)
    if new >= old:
        while k + 1 < n and buf[k + 1] < new:
            buf[k] = buf[k + 1]
            k += 1
    else:
        while k > 0 and buf[k - 1] > new:
            buf[k] = buf[k - 1]
            k -= 1
    buf[k] = new


@jit(fastmath=True, nopython=True, cache=True)
def _median_mad_sorted(s):
    """Median and median absolute deviation of the sorted array ``s`` in O(n)"""
    n = len(s)
    if n % 2 == 1:
        med = s[n // 2]
    else:
        med = (s[n // 2 - 1] + s[n // 2]) / 2
    # |s - med| are two ascending runs: A[i] = med - s[r-1-i] (left of the median,
    # walking outward) and B[j] = s[r+j] - med (right). The MAD is the median of their
    # union: find the K smallest by binary search over how many come from A (O(log n))
    r = np.searchsorted(s, med)
    na = r
    nb = n - r
    K = (n - 1) // 2 + 1  # count up to the (lower) middle element
    lo = max(0, K - nb)
    hi = min(K, na)
    inf = np.inf
    v1 = 0.0
    v2 = 0.0
    while lo <= hi:
        i = (lo + hi) // 2
        j = K - i
        a_left = med - s[r - i] if i > 0 else -inf  # A[i-1]
        a_right = med - s[r - 1 - i] if i < na else inf  # A[i]
        b_left = s[r + j - 1] - med if j > 0 else -inf  # B[j-1]
        b_right = s[r + j] - med if j < nb else inf  # B[j]
        if a_left > b_right:
            hi = i - 1
        elif b_left > a_right:
            lo = i + 1
        else:
            v1 = max(a_left, b_left)  # K-th smallest
            v2 = min(a_right, b_right) if n % 2 == 0 else v1  # (K+1)-th for even n
            break
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
        else:  # windows only move forward: swap leaving for entering values
            j_out = cur_lo
            j_in = cur_hi
            while True:
                while j_out < lo and mask[j_out] <= 0:
                    j_out += 1
                while j_in < hi and mask[j_in] <= 0:
                    j_in += 1
                if j_out < lo and j_in < hi:
                    _replace(buf, n, flux[j_out], flux[j_in])
                    j_out += 1
                    j_in += 1
                elif j_out < lo:
                    n = _remove(buf, n, flux[j_out])
                    j_out += 1
                elif j_in < hi:
                    n = _insert(buf, n, flux[j_in])
                    j_in += 1
                else:
                    break
        cur_lo = lo
        cur_hi = hi
        if n == 0:
            continue  # stays NaN
        if method_code == 5:  # median: a lookup in the sorted buffer, no MAD needed
            if n % 2 == 1:
                out[i] = buf[n // 2]
            else:
                out[i] = (buf[n // 2 - 1] + buf[n // 2]) / 2
            continue
        med, mad = _median_mad_sorted(buf[:n])
        if mad == 0:
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
    idx_start, idx_end = _window_bounds(time, half_window)

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

    # mean: running sums (prefix sums of the unmasked flux and of the counts), O(1)
    # per point. Sums relative to a reference value keep the round-off small.
    if method_code == 6:
        ref = flux[0]
        csum = np.zeros(size + 1)
        ccount = np.zeros(size + 1)
        for j in range(size):
            use = mask[j] > 0
            csum[j + 1] = csum[j] + (flux[j] - ref if use else 0.0)
            ccount[j + 1] = ccount[j] + (1.0 if use else 0.0)
        for i in prange(size):
            if time[i] >= low_index and time[i] <= hi_index:
                count = ccount[idx_end[i]] - ccount[idx_start[i]]
                if count > 0:
                    mean_all[i] = ref + (csum[idx_end[i]] - csum[idx_start[i]]) / count
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
