from __future__ import print_function, division
import numpy as np
from numba import jit


@jit(fastmath=True, nopython=True, cache=True)
def lowess(x, y, mask, window_length, maxiter=30, ftol=1e-6):

    def calc_x_weights(x, xi, y_weights):
        radius = max(xi - x[0], x[-2] - xi)
        x_weights = (1 - (np.abs(x - xi) / radius) ** 3) ** 3  # tricube
        if np.any(y_weights):
            x_weights = x_weights * y_weights
        return x_weights / np.nansum(x_weights)

    def calc_y_weights(y, mask, trend):
        diff = np.abs(y - trend)
        diff /= 6 * np.nanmedian(diff) + 1e-100  # avoid division by zero if median == 0
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

    points = len(x)
    y_fit_previous = np.ones(points)
    y_weights = np.zeros(points)
    for iter in range(maxiter):
        left = 0
        if (np.max(x) - np.min(x)) == 0:
            return np.full(len(x), np.nan)
        right = int((window_length / (np.max(x) - np.min(x))) * points)
        trend = np.zeros(points)
        for i in range(points):
            while right < points and (x[i] > (x[left] + x[right]) / 2):
                left += 1  # move the window forward
                right += 1
            x_weights = calc_x_weights(x[left:right], x[i], y_weights[left:right])
            trend[i] = calc_y_fit(x[left:right], y[left:right], x[i], x_weights)
        y_weights = calc_y_weights(y, mask, trend)
        if np.max(np.abs(y_fit_previous - trend)) < ftol:  # Check convergence
            break
        y_fit_previous = trend
    return trend
