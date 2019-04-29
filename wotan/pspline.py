from __future__ import print_function, division
import numpy
import wotan.constants as constants
from wotan.helpers import cleaned_array


def pspline(time, flux):
    try:
        from pygam import LinearGAM, s
    except:
        raise ImportError('Could not import pygam')

    newflux = flux.copy()
    newtime = time.copy()
    detrended_flux = flux.copy()

    for i in range(constants.PSPLINES_MAXITER):
        mask_outliers = numpy.ma.where(
            1-detrended_flux < constants.PSPLINES_STDEV_CUT*numpy.std(detrended_flux))
        newtime, newflux = cleaned_array(newtime[mask_outliers], newflux[mask_outliers])
        gam = LinearGAM(s(0, n_splines=constants.PSPLINES_MAX_SPLINES))
        search_gam = gam.gridsearch(newtime[:, numpy.newaxis], newflux, progress=False)
        trend = search_gam.predict(newtime)
        detrended_flux = newflux / trend
        stdev = numpy.std(detrended_flux)
        mask_outliers = numpy.ma.where(
            1-detrended_flux > constants.PSPLINES_STDEV_CUT*numpy.std(detrended_flux))
        print('Iteration:', i + 1, 'Rejected outliers:', len(mask_outliers[0]))

        # Check convergence
        if len(mask_outliers[0]) == 0:
            print('Converged.')
            break

    # Final iteration, applied to unclipped time series (interpolated over clipped values)
    mask_outliers = numpy.ma.where(1-detrended_flux < constants.PSPLINES_STDEV_CUT*stdev)
    newtime, newflux = cleaned_array(newtime[mask_outliers], newflux[mask_outliers])
    gam = LinearGAM(s(0, n_splines=constants.PSPLINES_MAX_SPLINES))
    search_gam = gam.gridsearch(newtime[:, numpy.newaxis], newflux, progress=False)
    trend = search_gam.predict(time)

    return trend
