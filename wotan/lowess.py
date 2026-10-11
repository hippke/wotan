from __future__ import print_function, division
import numpy as np
from numba import jit


@jit(fastmath=True, nopython=True, cache=True)
def _local_linear_fit(x, y, y_weights, imin, imax, xi, radius):
    """Weighted linear fit of y(x) over x[imin:imax], evaluated at ``xi``, with the
    weights y_weights * tricube(|x - xi| / radius). One pass over the window (no
    temporary arrays); the fit is centred on the weighted mean of x for accuracy.
    Equals the "projection vector" form of statsmodels."""
    # first pass: weighted mean of x (the normalisation of the weights)
    sw = 0.0
    swx = 0.0
    for j in range(imin, imax):
        w = y_weights[j] * (1 - (np.abs(x[j] - xi) / radius) ** 3) ** 3  # tricube
        sw += w
        swx += w * x[j]
    xm = swx / sw
    # second pass: weighted variance of x and covariance of x and y, about xm
    sxx = 0.0
    sxy = 0.0
    sy = 0.0
    for j in range(imin, imax):
        w = y_weights[j] * (1 - (np.abs(x[j] - xi) / radius) ** 3) ** 3
        dx = x[j] - xm
        sxx += w * dx * dx
        sxy += w * dx * y[j]
        sy += w * y[j]
    if sxx == 0:
        return np.nan
    return sy / sw + (xi - xm) * sxy / sxx


@jit(fastmath=True, nopython=True, cache=True)
def lowess(x, y, mask, window_length, cval=6.0, maxiter=30, ftol=1e-6):
    """Locally weighted linear regression (Cleveland 1979) with a window in units of
    ``x``: the trend at ``x[i]`` uses all points with ``|x - x[i]| < window_length/2``,
    weighted by the tricube kernel (radius ``window_length/2``). The robustness weights
    are bisquare, with cut-off ``cval`` times the median absolute residual
    (``cval = 6`` in Cleveland 1979). Masked points have weight zero."""

    def calc_y_weights(y, mask, trend):
        diff = np.abs(y - trend)
        diff /= cval * np.nanmedian(diff) + 1e-100  # avoid division by zero if median == 0
        diff[diff > 1] = 1
        y_weights = (1 - diff**2)**2
        y_weights[mask == 0] = 0
        return y_weights  # bisquare

    points = len(x)
    if (np.max(x) - np.min(x)) == 0:
        return np.full(points, np.nan)
    # Window borders in time (x is sorted): all points with |x - x[i]| < window/2
    radius = window_length / 2
    indices_l = np.searchsorted(x, x - radius)
    indices_r = np.searchsorted(x, x + radius)

    # Convergence tolerance relative to the range of the data: scale and offset free
    ftol = ftol * (np.max(y) - np.min(y))
    y_fit_previous = np.ones(points)
    y_weights = np.ones(points)
    y_weights[mask == 0] = 0
    trend = np.zeros(points)
    for iter in range(maxiter):
        trend = np.zeros(points)
        for i in range(points):
            imin = indices_l[i]
            imax = indices_r[i]
            trend[i] = _local_linear_fit(x, y, y_weights, imin, imax, x[i], radius)
        y_weights = calc_y_weights(y, mask, trend)
        if np.max(np.abs(y_fit_previous - trend)) < ftol:  # Check convergence
            break
        y_fit_previous = trend
    return trend
