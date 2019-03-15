import numpy
import numba


@numba.jit(fastmath=True, parallel=False, nopython=True)
def biweight_location_iter(data, c=6.0, ftol=1e-6, maxiter=15):
    """Robust location estimate using iterative Tukey's biweight"""

    delta_center = 1e100
    niter = 0

    # Initial estimate for the central location
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    center = center_old = numpy.median(data)
    if mad == 0:
        return center

    while not niter >= maxiter and not (abs(delta_center) <= ftol):
        # New estimate for center
        distance = data - center
        weight = distance / (c * mad)
        mask = (numpy.abs(weight) >= 1)
        weight = (1 - weight ** 2) ** 2
        weight[mask] = 0
        center += numpy.sum(distance * weight) / numpy.sum(weight)

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center  
        center_old = center
        niter += 1
    return center


@numba.jit(nopython=True)
def roll(array, shift):
    n = array.size
    shift %= n
    indexes = numpy.concatenate((numpy.arange(n - shift, n), numpy.arange(n - shift)))
    res = array.take(indexes)
    return res


@numba.jit(fastmath=True, parallel=False, nopython=True)
def running_segment(t, y, window, c=6, ftol=1e-6, maxiter=15):
    mean_all = numpy.full(len(t), numpy.nan)
    for i in range(len(t)-1):
        if t[i] > numpy.min(t) + window / 2 and t[i] < numpy.max(t)-(window / 2):
            mean_segment = numpy.nan
            idx_start = numpy.argmax(t > t[i] - window/2)# - 1
            idx_end = numpy.argmax(t > t[i] + window/2)# - 1
            data_segment = y[idx_start:idx_end]
            mean_segment = biweight_location_iter(
                data_segment,
                c=c,
                ftol=ftol,
                maxiter=maxiter
                )
            mean_all[i+int(window/2)] = mean_segment
    mean_all = roll(mean_all, int(window/2))
    return mean_all


def get_gaps_indexes(t, y, window):
    gaps = numpy.diff(t)
    gaps_indexes = numpy.where(gaps > window)
    gaps_indexes = numpy.add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = numpy.concatenate(gaps_indexes).ravel()
    gaps_indexes = numpy.append(numpy.array([0]), gaps_indexes)  # Start
    gaps_indexes = numpy.append(gaps_indexes, numpy.array([len(t)+1]))  # End point
    return gaps_indexes



def trendi(t, y, window, gaps_indexes, c=6, ftol=1e-6, maxiter=15):
    #y_new = numpy.array([])
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
        #trend = trend + trend_segment
        trend = numpy.append(trend, trend_segment)# = numpy.append(trend, trend_segment)
    print(len(trend))
    return trend



def running_biweight(t, y, window, c=6, ftol=1e-6, maxiter=15):
    gaps_indexes = get_gaps_indexes(t, y, window)
    trend = trendi(t, y, window, gaps_indexes, c=6, ftol=1e-6, maxiter=15)
    #trend = numpy.array(trend)
    #trend = numpy.concatenate(trend).ravel()

    print(trend)
    #print(gaps_indexes)
    
    return trend
