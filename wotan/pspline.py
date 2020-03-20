from __future__ import print_function, division
import numpy as np
import wotan.constants as constants
from wotan.helpers import cleaned_array


def pspline(time, flux, edge_cutoff, max_splines, return_nsplines, verbose):
    try:
        from pygam import LinearGAM, s
    except:
        raise ImportError('Could not import pygam')

    newflux = flux.copy()
    newtime = time.copy()
    detrended_flux = flux.copy()/np.nanmedian(newflux)

    for i in range(constants.PSPLINES_MAXITER):
        mask_outliers = np.ma.where(
            np.abs(1-detrended_flux) < constants.PSPLINES_STDEV_CUT*np.std(detrended_flux))
        newtime, newflux = cleaned_array(newtime[mask_outliers], newflux[mask_outliers])
        gam = LinearGAM(s(0, n_splines=max_splines))
        search_gam = gam.gridsearch(newtime[:, np.newaxis], newflux, progress=False)
        trend = search_gam.predict(newtime)
        detrended_flux = newflux / trend
        stdev = np.std(detrended_flux)
        mask_outliers = np.ma.where(
            np.abs(1-detrended_flux) > constants.PSPLINES_STDEV_CUT*np.std(detrended_flux))
        if verbose:
            print('Iteration:', i + 1, 'Rejected outliers:', len(mask_outliers[0]))
            # Check convergence
            if len(mask_outliers[0]) == 0:
                print('Converged.')
                break

    # Final iteration, applied to unclipped time series (interpolated over clipped values)
    mask_outliers = np.ma.where(np.abs(1-detrended_flux) < constants.PSPLINES_STDEV_CUT*stdev)
    newtime, newflux = cleaned_array(newtime[mask_outliers], newflux[mask_outliers])
    gam = LinearGAM(s(0, n_splines=max_splines))
    search_gam = gam.gridsearch(newtime[:, np.newaxis], newflux, progress=False)
    trend = search_gam.predict(time)

    # Cut off edges
    if edge_cutoff > 0:
        low_index = np.argmax(time > (min(time)+edge_cutoff))
        hi_index = np.argmax(time > (max(time)-edge_cutoff))
        trend[:low_index] = np.nan
        trend[hi_index:] = np.nan

    nsplines = np.ceil(gam.statistics_['edof'])
    return trend, nsplines
