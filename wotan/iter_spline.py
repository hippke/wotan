from __future__ import print_function, division
import numpy as np
import wotan.constants as constants
from wotan.helpers import cleaned_array
from scipy.interpolate import LSQUnivariateSpline


def iter_spline(time, flux, mask, window_length):
    masked_flux = flux[mask==1]
    masked_time = time[mask==1]
    no_knots = int((max(time) - min(time)) / window_length)
    newflux = masked_flux.copy()
    newtime = masked_time.copy()
    newtime, newflux = cleaned_array(newtime, newflux)
    detrended_flux = masked_flux.copy()
    for i in range(constants.PSPLINES_MAXITER):
        outliers = 1 - detrended_flux < constants.PSPLINES_STDEV_CUT * np.nanstd(detrended_flux)
        mask_outliers = np.ma.where(outliers)
        newtime, newflux = cleaned_array(newtime[mask_outliers], newflux[mask_outliers])
        # knots must not be at the edges, so we take them as [1:-1]
        if len(newtime) < 5:
            return np.full(len(time), np.nan)
        knots = np.linspace(min(newtime), max(newtime), no_knots)[1:-1]
        try:
            trend = LSQUnivariateSpline(newtime, newflux, knots)
        except:
            return np.full(len(time), np.nan)
        trend_segment = trend(newtime)
        detrended_flux = newflux / trend_segment
        outliers = 1 - detrended_flux > constants.PSPLINES_STDEV_CUT * np.nanstd(detrended_flux)
        mask_outliers = np.ma.where(mask_outliers)
        if len(mask_outliers[0]) == 0:
            break
    return trend(time)  # Final iteration: apply trend to clipped and masked values
