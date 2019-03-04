import numpy
import numba


@numba.jit(fastmath=True, parallel=False, nopython=True)
def roll(a, shift):
    n = a.size
    reshape = True
    if n == 0:
        return a
    shift %= n
    indexes = numpy.concatenate((numpy.arange(n - shift, n), numpy.arange(n - shift)))
    res = a.take(indexes)
    if reshape:
        res = res.reshape(a.shape)
    return res


@numba.jit(fastmath=True, parallel=False, nopython=True)
def biweight_location(data, c=6.0):
    M = numpy.median(data)
    d = data - M
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    if mad == 0:
        return M
    u = d / (c * mad)
    mask = (numpy.abs(u) >= 1)
    u = (1 - u ** 2) ** 2
    u[mask] = 0
    return M + (d * u).sum() / u.sum()


@numba.jit(fastmath=True, parallel=False, nopython=True)
def segment(t, y, window):
    mean = numpy.full(len(t), numpy.nan)
    for i in range(len(t)-1):
        if t[i] > numpy.min(t) + window / 2 and t[i] < numpy.max(t)-(window / 2):
            mean_segment = numpy.nan
            l = numpy.argmax(t > t[i] - window/2)
            h = numpy.argmax(t > t[i] + window/2)
            mean_segment = biweight_location(y[l:h])
            mean[i+int(window/2)] = mean_segment
    return roll(mean, int(window/2))


def detrend_biweight(t, y, window):
    gaps = numpy.diff(t)
    gaps_indexes = numpy.where(gaps > window)
    gaps_indexes = numpy.add(gaps_indexes, 1)  # Off by one :-)
    gaps_indexes = numpy.concatenate(gaps_indexes).ravel()
    gaps_indexes = numpy.append(numpy.array([0]), gaps_indexes)  # Start point: 0
    gaps_indexes = numpy.append(gaps_indexes, numpy.array([len(t)+1]))  # End point
    print(gaps_indexes)
    trend = numpy.array([])
    for i in range(len(gaps_indexes)-1):
        l = gaps_indexes[i]
        h = gaps_indexes[i+1]
        trend = numpy.append(trend, segment(t[l:h], y[l:h], window))
    return trend
