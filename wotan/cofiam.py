from __future__ import print_function, division
from numba import jit
import numpy as np
import wotan.constants as constants


@jit(fastmath=True, nopython=True, cache=True)
def matrix_gen_recurrence(t, degree):
    dur = 2 * (np.max(t) - np.min(t))
    # Work with time relative to the first point: keeps the design matrix well
    # conditioned for large time offsets (e.g. BJD)
    t0 = np.min(t)
    rows = len(t)
    cols = 2 * (degree + 1)
    matrix = np.ones(shape=(rows, cols))
    for x in range(rows):
        # sin/cos of y * val1 for y = 1, 2, ... by the angle-addition recurrence
        # (one sin and cos per row instead of one per element)
        val1 = (2 * np.pi * (t[x] - t0)) / dur
        s1 = np.sin(val1)
        c1 = np.cos(val1)
        s = s1
        c = c1
        for y in range(1, int(cols / 2)):
            matrix[x, y * 2] = s
            matrix[x, y * 2 + 1] = c
            s, c = s * c1 + c * s1, c * c1 - s * s1
        matrix[x, 1] = t[x] - t0
    return matrix


@jit(fastmath=True, nopython=True, cache=True)
def matrix_gen(t, degree):
    """Design matrix [1, t, sin, cos, ...] with sin and cos evaluated for every
    element (used by cofiam: it selects the model by the Durbin-Watson statistic, a
    discontinuous choice, so round-off level changes such as those of
    ``matrix_gen_recurrence`` can change the selected model)."""
    dur = 2 * (np.max(t) - np.min(t))
    # Work with time relative to the first point: keeps the design matrix well
    # conditioned for large time offsets (e.g. BJD)
    t0 = np.min(t)
    rows = len(t)
    cols = 2 * (degree + 1)
    matrix = np.ones(shape=(rows, cols))
    for x in range(rows):
        for y in range(1, int(cols / 2)):
            val = (2 * np.pi * (t[x] - t0) * y) / dur
            matrix[x, y * 2] = np.sin(val)
            matrix[x, y * 2 + 1] = np.cos(val)
        matrix[x, 1] = t[x] - t0
    return matrix


def detrend_cosine(t, y, window_length, robust, mask, verbose=False):
    degree = (int((max(t) - min(t)) / window_length))

    # robust version: sigma-clip flux from trend and iterate until convergence
    no_clip_previous = np.inf
    if not robust:
        converged = True
    else:
        converged = False
    matrix = matrix_gen_recurrence(t, degree)
    for i in range(constants.PSPLINES_MAXITER):
        # Add weights in order to weight down the masked values
        # Solution from https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix
        Aw = matrix * mask[:,np.newaxis]  # if real weights: sqrt
        Bw = y * mask  # if real weights: sqrt
        trend = np.matmul(matrix, np.linalg.lstsq(Aw, Bw, rcond=None)[0])
        detrended_flux = y / trend
        # Clip outliers on both sides of the trend (transits and flares). The scale is
        # the robust (MAD-based) sigma of the points still in use, so neither the
        # outliers nor the masked points inflate the threshold.
        used = mask > 1e-6
        sigma = 1.4826 * np.median(np.abs(detrended_flux[used] - np.median(detrended_flux[used])))
        if sigma > 0:
            mask_outliers = np.ma.where(
                np.abs(1 - detrended_flux) > constants.PSPLINES_STDEV_CUT * sigma)
        else:
            mask_outliers = np.ma.where(np.zeros(len(y), dtype=bool))
        mask[mask_outliers] = 1e-10
        if no_clip_previous == len(mask_outliers[0]):
            converged = True
        no_clip_previous = len(mask_outliers[0])
        if converged:
            if verbose:
                print('Converged.')
            break
        elif verbose:
            print('Iteration:', i + 1, 'Rejected outliers (total):', len(mask_outliers[0]))
    return trend


def detrend_cofiam(t, y, window_length, mask=None):
    """Cosine Filtering with Autocorrelation Minimization (Kipping et al. 2013):
    fit 1, 2, ... harmonics and keep the model with the best Durbin-Watson statistic."""
    degree = (int((max(t) - min(t)) / window_length))
    if mask is None:
        mask = np.ones(len(t))
    dw_previous = np.inf
    dw_mask = mask > 0
    trend_previous = None
    for k_m in range(1, degree + 1):
        matrix = matrix_gen(t, k_m)
        trend = np.matmul(matrix, np.linalg.lstsq(
            matrix * mask[:, np.newaxis], y * mask, rcond=-1)[0])

        # Durbin-Watson autocorrelation statistics
        dw_y = y[dw_mask] / trend[dw_mask] - 1
        dw = np.abs(np.sum((dw_y[1:] - dw_y[:-1]) ** 2) / (np.sum(dw_y ** 2)) - 2)

        # If Durbin-Watson *increased* this round: Previous was the best
        if dw > dw_previous:
            return trend_previous
        dw_previous = dw
        trend_previous = trend
    if trend_previous is None:  # degree 0: constant + linear term only
        matrix = matrix_gen(t, 0)
        trend_previous = np.matmul(matrix, np.linalg.lstsq(
            matrix * mask[:, np.newaxis], y * mask, rcond=-1)[0])
    return trend_previous
