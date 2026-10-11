from __future__ import print_function, division
from numba import jit
import numpy
from numpy import mean, median, inf, sin, exp, pi, array, sort
import wotan.constants as constants


@jit(fastmath=True, nopython=True, cache=True)
def location_iter(data, cval, method_code):
    """Robust location estimators"""

    # Numba can't handle strings, so we're passing the location estimator as an int:
    # 1 : biweight
    # 2 : andrewsinewave
    # 3 : welsch
    # (the others are not relevant for location_iter)

    # Initial estimate for the central location
    delta_center = inf
    median_data = median(data)
    mad = median(numpy.abs(data - median_data))
    center = center_old = median_data

    # Neglecting this case was a bug in scikit-learn
    if mad == 0:
        return center

    # one expensive division here, instead of two per loop later
    cmad = 1 / (cval * mad)

    # Convergence tolerance relative to the scale of the data (the MAD), so that the
    # estimate does not depend on the flux unit or offset
    tol = constants.FTOL * mad
    n_iter = 0

    # Newton-Raphson iteration, where each result is taken as the initial value of the
    # next iteration. Stops when the difference of a round is below ``FTOL`` threshold
    # One fused pass per iteration (no temporary arrays): accumulate sum(d*w) and sum(w)
    n = len(data)
    while numpy.abs(delta_center) > tol and n_iter < 1000:
        n_iter += 1
        sum_dw = 0.0
        sum_w = 0.0
        for j in range(n):
            distance = data[j] - center
            dmad = distance * cmad
            # biweight
            if method_code == 1:
                if numpy.abs(dmad) >= 1:
                    continue
                weight = (1 - dmad ** 2) ** 2
            # andrewsinewave
            elif method_code == 2:
                if dmad == 0:
                    dmad = 1e-10  # avoid division by zero
                if numpy.abs(dmad) >= pi:
                    continue
                weight = sin(dmad) / dmad
            # welsch
            else:
                if numpy.abs(dmad) >= 1:
                    continue
                weight = exp(-(dmad ** 2) / 2)
            sum_dw += distance * weight
            sum_w += weight
        center += sum_dw / sum_w

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center
        center_old = center
    return center


@jit(fastmath=True, nopython=True, cache=True)
def tau(x, cval):
    """Robust estimators of location and scale, with breakdown points of 50%.
    Also referred to as: Tau measure of location by Yohai and Zamar
    Source: Yohai and Zamar JASA, vol 83 (1988), pp 406-413 and 
            Maronna and Zamar Technometrics, vol 44 (2002), pp. 307-317"""

    med = median(x)
    mad = median(numpy.abs(x - med))
    if mad == 0:
        return med
    zscore = 0.6745  # Z-score of the 75th percentile of the normal distribution
    s = mad / zscore  # normal-consistent scale
    wnom = 0
    wden = 0
    for i in range(len(x)):
        y = (x[i] - med) / s
        # Biweight-type weight, zero for outliers beyond ``cval`` scale units
        if abs(y) < cval:
            temp = (1 - (y / cval)**2)**2
            wnom += temp * x[i]
            wden += temp
    if wden == 0:
        return med
    return wnom / wden


@jit(fastmath=True, nopython=True, cache=True)
def huber_psi(x, cval):
    """One-step M-estimator of location using Huber's psi"""
    med = median(x)
    mad = median(numpy.abs(x - med))
    if mad == 0:
        return med
    y = (x - med) / mad
    b = (numpy.abs(y) <= cval).sum()  # number of unclipped values (before clipping!)
    if b == 0:
        return med
    y[y > cval] = cval
    y[y < -cval] = -cval
    a = numpy.sum(y)
    return med + mad * a / b


@jit(fastmath=True, nopython=True, cache=True)
def hampelfilt(data, cval):
    """Values beyond (cval * mad {=median absolute deviation}) are replaced with the 
    median. Source: Ronald K. Pearson, Moncef Gabbouj (p. 147), "Nonlinear Digital 
    Filtering with Python: An Introduction" recommend cval=3"""
    
    flux = data.copy()
    median_data = median(flux)
    diff = numpy.abs(flux - median_data)
    mad = median(diff)
    flux[diff >= cval * mad] = median_data

    return mean(flux)


@jit(fastmath=True, nopython=True, cache=True)
def trim_mean(data, proportiontocut):
    """Mean of array after trimming `proportiontocut` from both tails."""
    len_data = len(data)
    sorted_data = sort(data)
    cut_idx = int(len_data * proportiontocut)
    return mean(sorted_data[cut_idx:len_data-cut_idx])


@jit(fastmath=True, nopython=True, cache=True)
def winsorize_mean(data, proportiontocut):
    """Mean of array after winsorizing `proportiontocut` from both tails."""
    sorted_data = sort(data)
    n = len(sorted_data)
    k = int(proportiontocut * n)  # number of values replaced in each tail
    if k > 0:
        sorted_data[:k] = sorted_data[k]
        sorted_data[n - k:] = sorted_data[n - k - 1]
    return mean(sorted_data)


@jit(fastmath=True, nopython=True, cache=True)
def hodges(data):
    """Hodges-Lehmann-Sen robust location estimator"""
    i = 0
    j = 0
    len_data = len(data)
    hodges = []
    while i < len_data:
        while j < len_data:
            hodges.append((data[i] + data[j]) / 2)
            j += 1
        i += 1
        j = i
    return median(array(hodges))
