from __future__ import print_function, division
import numpy as np
import wotan.constants as constants
from wotan.helpers import cleaned_array


def pspline(
    time, 
    flux, 
    edge_cutoff, 
    max_splines, 
    stdev_cut, 
    return_nsplines, 
    verbose,
    mask=None
    ):
    try:
        from pygam import LinearGAM, s
    except ImportError:
        raise ImportError('Could not import pygam')

    # The grid search refits the GAM for every smoothing parameter on the same
    # points, and pygam rebuilds the (identical) B-spline model matrix every time.
    # Cache it per set of points and basis (the smoothing parameter is not part of it).
    model_matrices = {}

    class CachedLinearGAM(LinearGAM):
        def _modelmat(self, X, term=-1):
            X_arr = np.asarray(X)
            key = (term, X_arr.shape, X_arr.tobytes(), repr(self.edge_knots_),
                   repr([(t.n_splines, t.spline_order) for t in self.terms
                         if hasattr(t, "n_splines")]))
            if key not in model_matrices:
                model_matrices.clear()  # keep only the current points
                model_matrices[key] = super()._modelmat(X, term=term)
            return model_matrices[key].copy()

    if mask is None:
        mask = np.ones(len(time))
    # Fit on the unmasked points only; the trend is evaluated at all times
    all_time = time
    newflux = flux[mask > 0].copy()
    newtime = time[mask > 0].copy()
    if len(newtime) == 0:
        return np.full(len(all_time), np.nan), np.nan
    detrended_flux = newflux / np.nanmedian(newflux)

    for i in range(constants.PSPLINES_MAXITER):
        mask_outliers = np.ma.where(
            np.abs(1 - detrended_flux) < stdev_cut * np.std(detrended_flux)
        )
        if len(mask_outliers[0]) != 0:  # Do not cut if zero points remain
            newtime, newflux = cleaned_array(
                newtime[mask_outliers], newflux[mask_outliers]
            )
        gam = CachedLinearGAM(s(0, n_splines=max_splines))
        search_gam = gam.gridsearch(newtime[:, np.newaxis], newflux, progress=False)
        trend = search_gam.predict(newtime)
        detrended_flux = newflux / trend
        stdev = np.std(detrended_flux)
        mask_outliers = np.ma.where(
            np.abs(1 - detrended_flux) > stdev_cut * np.std(detrended_flux)
        )
        if verbose:
            print("Iteration:", i + 1, "Rejected outliers:", len(mask_outliers[0]))
        # Check convergence
        if len(mask_outliers[0]) == 0:
            if verbose:
                print("Converged.")
            break

    # Final iteration, applied to unclipped time series (interpolated over clipped values)
    mask_outliers = np.ma.where(np.abs(1 - detrended_flux) < stdev_cut * stdev)
    if len(mask_outliers[0]) != 0:  # Do not cut if zero points remain
        newtime, newflux = cleaned_array(newtime[mask_outliers], newflux[mask_outliers])
    gam = CachedLinearGAM(s(0, n_splines=max_splines))
    search_gam = gam.gridsearch(newtime[:, np.newaxis], newflux, progress=False)
    trend = search_gam.predict(all_time)
    time = all_time

    # Cut off edges
    if edge_cutoff > 0:
        low_index = np.argmax(time > (min(time) + edge_cutoff))
        hi_index = np.argmax(time > (max(time) - edge_cutoff))
        trend[:low_index] = np.nan
        trend[hi_index:] = np.nan

    nsplines = np.ceil(gam.statistics_["edof"])
    return trend, nsplines
