import numpy
import numba
from numpy import pi, sin, exp

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
def biweight_location(data, c=4.685):  # 6 default; 4.685 for 95% efficiency
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
#https://cran.r-project.org/web/packages/robustbase/vignettes/psi_functions.pdf
def welsh_location(data, c=2.11):
    M = numpy.median(data)
    d = data - M
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    if mad == 0:
        return M
    u = d / (c * mad)
    mask = (numpy.abs(u) >= 1)
    u = exp(-u**2 / 2)
    u[mask] = 0
    return M + (d * u).sum() / u.sum()


@numba.jit(fastmath=True, parallel=False, nopython=True)
def andrewsinewave_location(data, c=1.339):
    M = numpy.median(data)
    d = data - M
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    if mad == 0:
        return M
    u = d / (c * mad)
    u[u==0] = 1e-100  # avoid division by zero
    mask = (numpy.abs(u) >= pi)
    u = sin(u) / u
    u[mask] = 0
    return M + (d).sum() / u.sum()

"""
data = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
print(biweight_location(data, c=4.685))
#print(biweight_location(data, c=4.685))
print(andrewsinewave_location(data, c=4.685))
print(welsh_location(data))
"""

@numba.jit(fastmath=True, parallel=False, nopython=True)
def segment(t, y, window):
    mean = numpy.full(len(t), numpy.nan)
    for i in range(len(t)-1):
        if t[i] > numpy.min(t) + window / 2 and t[i] < numpy.max(t)-(window / 2):
            mean_segment = numpy.nan
            l = numpy.argmax(t > t[i] - window/2)
            h = numpy.argmax(t > t[i] + window/2)
            #mean_segment = biweight_location(y[l:h])
            #mean_segment = andrewsinewave_location(y[l:h])
            mean_segment = welsh_location(y[l:h])
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
