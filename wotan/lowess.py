from __future__ import print_function, division
import numpy as np
from numba import jit


@jit(fastmath=True, nopython=True, cache=True)
def lowess(x, y, mask, window_length, cval, maxiter=30, ftol=1e-6):

    def calc_x_weights(x, xi, y_weights, window_length):
        radius = window_length/2
        x_weights = y_weights * (1 - (np.abs(x - xi) / radius) ** 3) ** 3  # tricube
        return x_weights / np.nansum(x_weights)

    def calc_y_weights(y, mask, trend):
        diff = np.abs(y - trend)
        diff /= cval * 1.4826 * np.nanmedian(diff) + 1e-100  # avoid division by zero if median == 0
        diff[diff > 1] = 1
        y_weights = (1 - diff**2)**2
        y_weights[mask == 0] = 0
        return y_weights  # bisquare

    def calc_y_fit(x, y, xi, weights):  # "projection vector" as in statsmodels
        w1 = np.sum(weights * x)  #
        w2 = np.sum(weights * (x - w1)**2)
        if w2 == 0:
            return np.nan
        else:
            return np.sum(weights * ((1 + (xi - w1) * (x - w1) / w2) * y))

    indices_l = np.searchsorted(x, x - window_length / 2)
    indices_r = np.searchsorted(x, x + window_length / 2)

    points = len(x)
    y_fit_previous = np.ones(points)
    y_weights = np.ones(points)
    y_weights[mask == 0] = 0
    for iter in range(maxiter):
        if (np.max(x) - np.min(x)) == 0:
            return np.full(len(x), np.nan)
        trend = np.zeros(points)
        for i in range(points):
            imin = indices_l[i]
            imax = indices_r[i]
            x_weights = calc_x_weights(x[imin:imax], x[i], y_weights[imin:imax], window_length)
            trend[i] = calc_y_fit(x[imin:imax], y[imin:imax], x[i], x_weights)
        y_weights = calc_y_weights(y, mask, trend)
        if np.max(np.abs(y_fit_previous - trend)) < ftol:  # Check convergence
            break
        y_fit_previous = trend
    return trend
