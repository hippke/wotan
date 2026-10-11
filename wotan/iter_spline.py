from __future__ import print_function, division
import numpy as np
import wotan.constants as constants
from wotan.helpers import cleaned_array
from scipy.interpolate import LSQUnivariateSpline


def iter_spline(time, flux, mask, window_length):
    masked_flux = flux[mask==1]
    masked_time = time[mask==1]
    no_knots = int((max(time) - min(time)) / window_length)
    all_time, all_flux = cleaned_array(masked_time.copy(), masked_flux.copy())
    keep = np.ones(len(all_time), dtype=bool)
    trend = None
    for i in range(constants.PSPLINES_MAXITER):
        newtime, newflux = all_time[keep], all_flux[keep]
        # knots must not be at the edges, so we take them as [1:-1]
        if len(newtime) < 5:
            return np.full(len(time), np.nan)
        knots = np.linspace(min(newtime), max(newtime), no_knots)[1:-1]
        try:
            trend = LSQUnivariateSpline(newtime, newflux, knots)
        except Exception:
            return np.full(len(time), np.nan)
        # Clip outliers on both sides of the trend (dips and flares), using the robust
        # (MAD-based) sigma of the residuals of *all* points
        detrended_flux = all_flux / trend(all_time)
        sigma = 1.4826 * np.nanmedian(np.abs(detrended_flux - np.nanmedian(detrended_flux)))
        new_keep = np.abs(1 - detrended_flux) < constants.PSPLINES_STDEV_CUT * sigma
        if sigma == 0 or np.array_equal(new_keep, keep):
            break  # converged: the clipped set does not change any more
        keep = new_keep
    return trend(time)  # Final iteration: apply trend to clipped and masked values
