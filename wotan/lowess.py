from __future__ import print_function, division
import numpy as np
from numba import jit, prange


@jit(fastmath=True, nopython=True, cache=True, error_model="numpy")
def _local_linear_fit(x, y, y_weights, imin, imax, xi, radius):
    """Weighted linear fit of y(x) over x[imin:imax], evaluated at ``xi``, with the
    normalised weights w = y_weights * tricube(|x - xi| / radius) / sum(...): the
    "projection vector" form of statsmodels,
        sum(w * (1 + (xi - w1) * (x - w1) / w2) * y),
    with w1 = sum(w * x) and w2 = sum(w * (x - w1)^2). Computed in passes over the
    window without temporary arrays, term by term as before."""
    n = imax - imin
    k = np.empty(n)  # kernel weights, computed once
    sw = 0.0
    for j in range(n):
        k[j] = y_weights[imin + j] * (1 - (np.abs(x[imin + j] - xi) / radius) ** 3) ** 3
        sw += k[j]
    w1 = 0.0
    for j in range(n):
        w1 += k[j] / sw * x[imin + j]
    # w2, and the two sums of the fit sum(w*y) + (xi-w1)/w2 * sum(w*(x-w1)*y), together
    w2 = 0.0
    swy = 0.0
    swdy = 0.0
    for j in range(n):
        w = k[j] / sw
        d = x[imin + j] - w1
        w2 += w * d ** 2
        swy += w * y[imin + j]
        swdy += w * d * y[imin + j]
    if w2 == 0:
        return np.nan
    fit = swy + (xi - w1) * swdy / w2
    return fit


@jit(fastmath=True, nopython=True, cache=True, parallel=True)
def lowess(x, y, mask, window_length, cval=6.0, maxiter=30, ftol=1e-6):
    """Locally weighted linear regression (Cleveland 1979) with a window in units of
    ``x``: the trend at ``x[i]`` uses all points with ``|x - x[i]| < window_length/2``,
    weighted by the tricube kernel (radius ``window_length/2``). The robustness weights
    are bisquare, with cut-off ``cval`` times the median absolute residual
    (``cval = 6`` in Cleveland 1979). Masked points have weight zero."""

    def calc_y_weights(y, mask, trend):
        diff = np.abs(y - trend)
        # Scale: cval * median residual, but not below the round-off level of the
        # data (noise-free data would otherwise down-weight points by their round-off
        # residuals, and the weights collapse); this also avoids a division by zero
        scale = max(cval * np.nanmedian(diff), 1e-12 * np.max(np.abs(y)) + 1e-100)
        diff /= scale
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
        for i in prange(points):  # the points are independent: parallel threads
            trend[i] = _local_linear_fit(
                x, y, y_weights, indices_l[i], indices_r[i], x[i], radius)
        y_weights = calc_y_weights(y, mask, trend)
        if np.max(np.abs(y_fit_previous - trend)) < ftol:  # Check convergence
            break
        y_fit_previous = trend
    return trend
