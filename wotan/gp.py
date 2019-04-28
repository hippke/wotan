from __future__ import print_function, division
import numpy
import wotan.constants as constants
from scipy.signal import lombscargle


def make_gp(time, flux, kernel, kernel_size, kernel_period):
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

    if kernel == 'periodic':
        if kernel_period is None:
            raise ValueError('kernel_period must be specified')
        if not isinstance(kernel_period, float) and not isinstance(kernel_period, int):
            raise ValueError('kernel_period must be a floating point value')
        if kernel_period <= 0 or kernel_period >= float("inf"):
            raise ValueError('kernel_period must be finite and positive')

    # GPs need flux near zero, otherwise they often fail to converge
    # So, we normalize by some constant (the median) and later transform back
    offset = numpy.median(flux)
    flux -= offset   

    # Determine most significant period
    if kernel == 'periodic_auto':
        time_span = numpy.max(time) - numpy.min(time)
        cadence = numpy.nanmedian(numpy.diff(time))
        freqs = numpy.geomspace(1/time_span, 1/cadence, constants.LS_FREQS)
        pgram = lombscargle(time, flux, freqs)
        kernel_period = 1 / freqs[numpy.argmax(pgram)] * 2 * numpy.pi

    # RBF and matern kernels are very similar when matern's (kernel_size * 1000)
    if kernel == 'matern':
        kernel_size *= 1000

    kernel_size_bounds = (0.5 * kernel_size, 2 * kernel_size)

    if kernel is None or kernel == 'squared_exp':
        use_kernel = RBF(kernel_size, kernel_size_bounds)
        grid = time.reshape(-1, 1)
        trend_segment = GaussianProcessRegressor(
            use_kernel, alpha=1e-5).fit(grid, flux).predict(grid)
    elif kernel == 'matern':
        use_kernel = Matern(kernel_size, kernel_size_bounds, nu=3/2)
        grid = time.reshape(-1, 1)
        trend_segment = GaussianProcessRegressor(
            use_kernel, alpha=1e-5).fit(grid, flux).predict(grid)
    elif 'periodic' in kernel:
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
        grid = time.reshape(-1, 1)
        trend_segment = GaussianProcessRegressor(
            use_kernel + use_kernel2, alpha=1e-5).fit(grid, flux).predict(grid)

    return (trend_segment + offset)
