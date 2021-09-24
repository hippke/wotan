from __future__ import print_function, division
from numba import jit
import numpy as np
import wotan.constants as constants


@jit(fastmath=True, nopython=True, cache=True)
def matrix_gen(t, degree):
    dur = 2 * (np.max(t) - np.min(t))
    rows = len(t)
    cols = 2 * (degree + 1)
    matrix = np.ones(shape=(rows, cols))
    for x in range(rows):
        for y in range(1, int(cols / 2)):
            val = (2 * np.pi * t[x] * y) / dur
            matrix[x, y * 2] = np.sin(val)
            matrix[x, y * 2 + 1] = np.cos(val)
        matrix[x, 1] = t[x]
    return matrix


def detrend_cosine(t, y, window_length, robust, mask):
    degree = (int((max(t) - min(t)) / window_length))
    #if not robust:
    #    constants.PSPLINES_MAXITER = 0
        #matrix = matrix_gen(t, degree)
        #trend = np.matmul(matrix, np.linalg.lstsq(matrix, y, rcond=-1)[0])

    # robust version: sigma-clip flux from trend and iterate until convergence
    no_clip_previous = np.inf
    if not robust:
        converged = True
    else:
        converged = False
    for i in range(constants.PSPLINES_MAXITER):
        matrix = matrix_gen(t, degree)
        # Add weights in order to weight down the masked values
        # Solution from https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix
        Aw = matrix * mask[:,np.newaxis]  # if real weights: sqrt
        Bw = y * mask  # if real weights: sqrt
        trend = np.matmul(matrix, np.linalg.lstsq(Aw, Bw, rcond=None)[0])
        detrended_flux = y / trend
        mask_outliers = np.ma.where(
            1-detrended_flux > constants.PSPLINES_STDEV_CUT*np.std(detrended_flux))
        mask[mask_outliers] = 1e-10
        if no_clip_previous == len(mask_outliers[0]):
            converged = True
        no_clip_previous = len(mask_outliers[0])
        if converged:
            print('Converged.')
            break
        else:
            print('Iteration:', i + 1, 'Rejected outliers (total):', len(mask_outliers[0]))
    return trend


def detrend_cofiam(t, y, window_length):
    degree = (int((max(t) - min(t)) / window_length))
    dw_previous = np.inf
    dw_mask = np.array([True] * len(t))
    for k_m in range(1, degree + 1):
        matrix = matrix_gen(t, degree)
        trend = np.matmul(matrix, np.linalg.lstsq(matrix, y, rcond=-1)[0])

        # Durbin-Watson autocorrelation statistics
        dw_y = y[dw_mask] / trend[dw_mask] - 1
        dw = np.abs(np.sum((dw_y[1:] - dw_y[:-1]) ** 2) / (np.sum(dw_y ** 2)) - 2)

        # If Durbin-Watson *increased* this round: Previous was the best
        if dw > dw_previous:
            return trend
        dw_previous = dw
    matrix = matrix_gen(t, degree)
    trend = np.matmul(matrix, np.linalg.lstsq(matrix, y, rcond=-1)[0])
    return trend
