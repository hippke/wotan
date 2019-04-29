from __future__ import print_function, division
import numpy
import wotan.constants as constants
from wotan.helpers import cleaned_array
from scipy import interpolate


def iter_spline(time, flux, window_length):
    no_knots = (max(time) - min(time)) / window_length
    newflux = flux.copy()
    newtime = time.copy()
    detrended_flux = flux.copy()
    for i in range(constants.PSPLINES_MAXITER):
        mask_outliers = numpy.ma.where(
            1 - detrended_flux < constants.PSPLINES_STDEV_CUT * numpy.std(detrended_flux))
        newtime, newflux = cleaned_array(newtime[mask_outliers], newflux[mask_outliers])
        # knots must not be at the edges, so we take them as [1:-1]
        knots = numpy.linspace(min(newtime), max(newtime), no_knots)[1:-1]
        s = interpolate.LSQUnivariateSpline(newtime, newflux, knots)
        trend_segment = s(newtime)
        detrended_flux = newflux / trend_segment
        mask_outliers = numpy.ma.where(
            1 - detrended_flux > constants.PSPLINES_STDEV_CUT * numpy.std(detrended_flux))
        print('Iteration:', i + 1, 'Rejected outliers:', len(mask_outliers[0]))
        # Check convergence
        if len(mask_outliers[0]) == 0:
            print('Converged.')
            break
    # Final iteration applied all data interpolated over clipped values
    trend_segment = s(time)
    return trend_segment
