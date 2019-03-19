import numpy
import numba


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def biweight_location_iter(data, c=6.0, ftol=1e-6, maxiter=15):
    """Robust location estimate using iterative Tukey's biweight"""

    delta_center = 1e100
    niter = 0

    # Initial estimate for the central location
    median_data = numpy.median(data)
    mad = numpy.median(numpy.abs(data - median_data))
    center = center_old = median_data

    if mad == 0:
        return center  # Neglecting this case was a bug in scikit-learn

    cmad = 1 / (c * mad)  # one expensive division here, instead of two per loop later
    while niter < maxiter and abs(delta_center) > ftol:  # New estimate for center
        distance = data - center
        weight = (1 - (distance * cmad) ** 2) ** 2  # inliers with Tukey's biweight
        weight[(numpy.abs(distance * cmad) >= 1)] = 0  # outliers as zero
        center += numpy.sum(distance * weight) / numpy.sum(weight)

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center
        center_old = center
        niter += 1
    return center


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def roll(array, shift):
    """Replacement of 1D numpy roll in numba for (much) faster speed"""
    n = array.size
    shift %= n
    indexes = numpy.concatenate((numpy.arange(n - shift, n), numpy.arange(n - shift)))
    res = array.take(indexes)
    return res


@numba.jit(fastmath=True, parallel=False, nopython=True, cache=True)
def running_segment(t, y, window, c=6, ftol=1e-6, maxiter=15):
    mean_all = numpy.full(len(t), numpy.nan)

    # Move border checks out of loop (reason: large speed gain)
    half_window = window / 2
    lo = numpy.min(t) + half_window
    hi = numpy.max(t) - half_window
    idx_start = 0
    idx_end = 0

    for i in range(len(t)-1):
        if t[i] > lo and t[i] < hi:
            # Nice style would be:
            # idx_start = numpy.argmax(t > t[i] - window/2)
            # idx_end = numpy.argmax(t > t[i] + window/2)
            # But that's too slow (factor 10). Instead, we write:
            while t[idx_start] < t[i] - half_window:
                idx_start +=1
            while t[idx_end] < t[i] + half_window:
                idx_end +=1
            
            mean_segment = biweight_location_iter(
                y[idx_start:idx_end],
                c=c,
                ftol=ftol,
                maxiter=maxiter
                )
            
            mean_all[i+int(half_window)] = mean_segment
    mean_all = roll(mean_all, int(half_window))
    return mean_all


def get_gaps_indexes(time, window):
    """Array indexes where (time) series is interrupted for longer than (window)"""
    gaps = numpy.diff(time)
    gaps_indexes = numpy.where(gaps > window)
    gaps_indexes = numpy.add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = numpy.concatenate(gaps_indexes).ravel()
    gaps_indexes = numpy.append(numpy.array([0]), gaps_indexes)  # Start
    gaps_indexes = numpy.append(gaps_indexes, numpy.array([len(time)+1]))  # End point
    return gaps_indexes


def running_biweight(t, y, window, c=6, ftol=1e-6, maxiter=15):
    gaps_indexes = get_gaps_indexes(t, window)
    trend = numpy.array([])
    trend_segment = numpy.array([])
    for i in range(len(gaps_indexes)-1):
        start_segment = gaps_indexes[i]
        end_segment = gaps_indexes[i+1]
        trend_segment = running_segment(
            t[start_segment:end_segment],
            y[start_segment:end_segment],
            window,
            c=c,
            ftol=ftol,
            maxiter=maxiter
            )
        trend = numpy.append(trend, trend_segment)
    return trend
