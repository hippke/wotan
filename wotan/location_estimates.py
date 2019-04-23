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

    # Newton-Raphson iteration, where each result is taken as the initial value of the
    # next iteration. Stops when the difference of a round is below ``FTOL`` threshold
    while numpy.abs(delta_center) > constants.FTOL:
        distance = data - center
        dmad = distance * cmad

        # Inlier weights
        # biweight
        if method_code == 1:
            weight = (1 - dmad ** 2) ** 2
        # andrewsinewave
        if method_code == 2:
            # avoid division by zero
            dmad[dmad == 0] = 1e-10
            weight = sin(dmad) / dmad
        # welsch
        if method_code == 3:
            weight = exp(-(dmad**2) / 2)

        # Outliers with weight zero
        # biweight or welsch
        if method_code == 1 or method_code == 3:
            weight[(numpy.abs(dmad) >= 1)] = 0
        # andrewsinewave
        else:
            weight[(numpy.abs(dmad) >= pi)] = 0

        center += numpy.sum(distance * weight) / numpy.sum(weight)

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center
        center_old = center
    return center


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
    idx = int(proportiontocut * len(data)) + 1
    if idx < 0:
        idx = 0
    sorted_data[:idx] = sorted_data[idx]
    sorted_data[-idx:] = sorted_data[-idx]
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
