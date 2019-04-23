from __future__ import print_function, division
from numba import jit
from scipy.optimize import leastsq
import numpy
from numpy import (
    mean,
    array,
    inf,
    sin,
    cos,
    pi,
    ones,
    zeros,
)
import wotan.constants as constants


@jit(fastmath=True, nopython=True, cache=True)
def cofiam_detrend_curve(t, fact, D_max, k_max):
    ph = 2 * pi * t / D_max
    detr = ones(len(t)) * fact[0]
    for k in range(1, k_max + 1):
        detr += fact[2 * k - 1] * sin(ph * k) + fact[
            2 * k
        ] * cos(ph * k)
    return detr


def find_detrending_for_region(t, y, ferr, k_max, D_max):
    def fit_func(fact):
        return (
            y - cofiam_detrend_curve(t, fact, D_max, k_max)
        ) / ferr

    def out_func(blub):
        return cofiam_detrend_curve(
            t, best_param, D_max, k_max
        )

    x0 = zeros(2 * k_max + 1)
    x0[0] = mean(y)
    best_param, param_fit_status = leastsq(
        fit_func, x0, ftol=constants.FTOL
    )
    result = cofiam_detrend_curve(
        t, best_param, D_max, k_max
    )
    return out_func


def detrend_cofiam(t, y, ferr, window_length):
    D_max = 2 * (max(t) - min(t))
    k_max = max(
        1,
        min(
            constants.COFIAM_MAX_SINES,
            int(D_max / window_length),
        ),
    )
    dw_previous = inf
    dw_mask = array([True] * len(t))
    for k_m in range(1, k_max + 1):
        detrend_func_temp = find_detrending_for_region(
            t, y, ferr, k_m, D_max
        )
        trend = detrend_func_temp(t)

        # Durbin-Watson autocorrelation statistics
        dw_y = y[dw_mask] / trend[dw_mask] - 1
        dw = numpy.abs(
            numpy.sum((dw_y[1:] - dw_y[:-1]) ** 2)
            / (numpy.sum(dw_y ** 2))
            - 2
        )

        # If Durbin-Watson *increased* this round: Previous was the best
        if dw > dw_previous:
            return trend
        detrend_func = detrend_func_temp
        dw_previous = dw
    return detrend_func(t)
