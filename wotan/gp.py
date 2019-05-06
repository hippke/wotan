from __future__ import print_function, division
import numpy
import wotan.constants as constants
from scipy.signal import lombscargle
from wotan.helpers import cleaned_array


def make_gp(time, flux, kernel, kernel_size, kernel_period, robust):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared
    except:
        raise ImportError('Could not import sklearn')

    if kernel_size is None:
        raise ValueError('kernel_size must be specified')
    if not isinstance(kernel_size, float) and not isinstance(kernel_size, int):
        raise ValueError('kernel_size must be a floating point value')
    if kernel_size <= 0 or kernel_size >= float("inf"):
        raise ValueError('kernel_size must be finite and positive')

    # GPs need flux near zero, otherwise they often fail to converge
    # So, we normalize by some constant (the median) and later transform back
    offset = numpy.median(flux)
    flux -= offset

    # RBF and matern kernels are similar when matern's (kernel_size * 1000)
    if kernel == 'matern':
        kernel_size *= 1000

    kernel_size_bounds = (0.5 * kernel_size, 2 * kernel_size)
    grid = time.reshape(-1, 1)

    if kernel is None or kernel == 'squared_exp':
        use_kernel = RBF(kernel_size, kernel_size_bounds)

    if kernel == 'matern':
        use_kernel = Matern(kernel_size, kernel_size_bounds, nu=3/2)

    # Single pass or iteratively clipped
    if kernel == 'matern' or kernel == 'squared_exp' or kernel is None:
        if robust:
            converged = False
            newflux = flux.copy()
            newtime = time.copy()
            detrended_flux = flux.copy()
            for i in range(constants.PSPLINES_MAXITER):
                # Flux must be ~1. First round may by ~0. Then correct: 
                if abs(numpy.median(detrended_flux)) < 0.5:
                    detrended_flux += 1
                mask_outliers = numpy.ma.where(
                    1-detrended_flux < constants.PSPLINES_STDEV_CUT * numpy.std(
                        detrended_flux))
                newtime, newflux = cleaned_array(
                    newtime[mask_outliers], newflux[mask_outliers])
                grid = newtime.reshape(-1, 1)
                GP = GaussianProcessRegressor(use_kernel).fit(grid, newflux)
                trend_segment = GP.predict(grid)
                detrended_flux = (newflux + offset) / (trend_segment + offset)
                mask_outliers = numpy.ma.where(
                    1-detrended_flux > constants.PSPLINES_STDEV_CUT * numpy.std(
                        detrended_flux))
                print('Iteration:', i + 1, 'Rejected outliers:', len(mask_outliers[0]))
                # Check convergence
                if converged or len(mask_outliers[0]) == 0:
                    print('Converged.')
                    break
            # Final iteration, applied to unclipped time series 
            # (interpolated over clipped values)
            trend_segment = GP.predict(time.reshape(-1, 1))
        else:
            trend_segment = GaussianProcessRegressor(
                use_kernel).fit(grid, flux).predict(grid)

    # Single pass; currently no iterative periodic kernel implemented
    if 'periodic' in kernel:
        # Determine most significant period
        if kernel == 'periodic_auto':
            time_span = numpy.max(time) - numpy.min(time)
            cadence = numpy.nanmedian(numpy.diff(time))
            freqs = numpy.geomspace(1/time_span, 1/cadence, constants.LS_FREQS)
            pgram = lombscargle(time, flux, freqs)
            kernel_period = 1 / freqs[numpy.argmax(pgram)] * 2 * numpy.pi
        else:
            if kernel_period is None:
                raise ValueError('kernel_period must be specified')
            if not isinstance(kernel_period, float) and not isinstance(kernel_period, int):
                raise ValueError('kernel_period must be a floating point value')
            if kernel_period <= 0 or kernel_period >= float("inf"):
                raise ValueError('kernel_period must be finite and positive')
        kernel_period_bounds=(0.5 * kernel_period, 2 * kernel_period)
        # The periodic part
        use_kernel = ExpSineSquared(
            kernel_size,
            kernel_period,
            kernel_size_bounds,
            kernel_period_bounds
            )
        # For additional trends
        use_kernel2 = RBF(kernel_size, kernel_size_bounds)  
        trend_segment = GaussianProcessRegressor(
            use_kernel + use_kernel2).fit(grid, flux).predict(grid)

    return (trend_segment + offset)
