from __future__ import print_function, division
import numpy as np
from numba import jit


@jit(fastmath=True, nopython=True, cache=True)
def lowess(x, y, frac, maxiter=10):
    ftol = 1e-6
    n = len(x)
    k =  int(frac * n + 1e-10)
    y_fit = np.zeros(n)
    y_fit_previous = np.ones(n)
    resid_weights = np.zeros(n)
    for iter in range(maxiter):
        l = 0
        r = k
        y_fit = np.zeros(n)
        for i in range(n):
            l, r = inscope(x, i, n, l, r)
            reg_ok, weights = calculate_weights(x, resid_weights, i, l, r, iter > 0)
            y_fit[i] = calculate_y_fit(x, y, i, y_fit, weights, l, r, reg_ok)
        resid_weights = calculate_residual_weights(y, y_fit)
        
        # Check convergence
        diff = np.max(np.abs(y_fit_previous-y_fit))
        y_fit_previous = y_fit
        print(iter, diff)
        if diff < ftol:
            print('Lowess converged to', ftol, 'in', iter, 'iterations')
            break
    return y_fit


@jit(fastmath=True, nopython=True, cache=True)
def inscope(x, i, n, l, r):
    while True:
        if r < n:
            if (x[i] > (x[l] + x[r]) / 2):
                l += 1
                r += 1
            else:
                break
        else:
            break
    return l, r


@jit(fastmath=True, nopython=True, cache=True)
def calculate_weights(x, resid_weights, i, l, r, use_resid_weights):
    weights = np.zeros(len(x))
    radius = fmax(x[i] - x[l], x[r-1] - x[i])
    weights[l:r] = np.abs(x[l:r] - x[i]) / radius
    reg_ok = True
    if use_resid_weights:
        weights[l:r] = tricube(weights[l:r]) * resid_weights[l:r]
    else:
        weights[l:r] = tricube(weights[l:r])
    sum_weights = np.sum(weights[l:r])
    if sum_weights <= 0 or (np.sum(weights[l:r] != 0) == 1):
        reg_ok = False
    else:
        weights[l:r] = weights[l:r] / sum_weights
    return reg_ok, weights


@jit(fastmath=True, nopython=True, cache=True)
def calculate_residual_weights(y, y_fit):
    std_resid = np.abs(y - y_fit)
    median = np.median(std_resid)
    if median == 0:
        std_resid[std_resid > 0] = 1
    else:
        std_resid /= 6 * median
    std_resid[std_resid >= 1] = 1
    return bisquare(std_resid)


@jit(fastmath=True, nopython=True, cache=True)
def calculate_y_fit(x, y, i, y_fit, weights, l, r, reg_ok):
    if not reg_ok:
        y_fit[i] = y[i]
    else:
        wx = 0
        wsx = 0
        for j in range(l, r):
            wx += weights[j] * x[j]
        for j in range(l, r):
            wsx += weights[j] * (x[j] - wx) ** 2
        for j in range(l, r):
            y_fit[i] += weights[j] * (1 + (x[i] - wx) * (x[j] - wx) / wsx) * y[j]
    return y_fit[i]


@jit(fastmath=True, nopython=True, cache=True)
def tricube(w):
    return (1 - w ** 3) ** 3


@jit(fastmath=True, nopython=True, cache=True)
def bisquare(x):
    return (1.0 - x**2)**2


@jit(fastmath=True, nopython=True, cache=True)
def fmax(x, y):
    return x if x >= y else y
