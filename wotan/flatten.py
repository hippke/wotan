"""Wotan is a free and open source algorithm to automagically remove stellar trends
from light curves for exoplanet transit detection.
"""

from __future__ import print_function, division
import numpy as np
from numpy import array, isnan, float64, append, full, where, nan, ones, inf, median
from scipy.signal import savgol_filter, medfilt

# wotan
import wotan.constants as constants
from wotan.cofiam import detrend_cofiam, detrend_cosine
from wotan.gp import make_gp
from wotan.huber_spline import detrend_huber_spline
from wotan.slider import running_segment, running_segment_slow
from wotan.gaps import get_gaps_indexes
from wotan.t14 import t14
from wotan.pspline import pspline
from wotan.iter_spline import iter_spline
from wotan.regression import regression
from wotan.lowess import lowess


def flatten(time, 
            flux,
            window_length=None,
            edge_cutoff=0,
            break_tolerance=None,
            cval=None,
            return_trend=False,
            method='biweight',
            kernel=None,
            kernel_size=None,
            kernel_period=None,
            proportiontocut=constants.PROPORTIONTOCUT,
            robust=False, 
            max_splines=constants.PSPLINES_MAX_SPLINES,
            return_nsplines=False,
            mask=None,
            verbose=False
            ):
    """
    ``flatten`` removes low frequency trends in time-series data.
        
    Parameters
    ----------
    time : array-like
        Time values
    flux : array-like
        Flux values for every time point
    window_length : float
        The length of the filter window in units of ``time`` (usually days), or in
        cadences (for cadence-based sliders ``savgol`` and ``medfilt``).
    method : string, default: ``biweight``
        Detrending method. Rime-windowed sliders: ``median``, ``biweight``, ``hodges``,
        ``tau``, ``welsch``, ``huber``, ``huber_psi``, ``andrewsinewave``, ``mean``,
        ``hampel``, ``ramsay``, ``trim_mean``, ``hampelfilt``, ``winsorize``. Cadence
        based slider: ``medfilt``. Splines: ``hspline``, ``rspline`, ``pspline``.
        Locally weighted scatterplot smoothing: ``lowess``. Savitzky-Golay filter:
        ``savgol``. Gaussian processes: ``gp``. Cosine Filtering with Autocorrelation
        Minimization: ``cofiam``.  Cosine fitting: ``cosine``, Friedman's Super-Smoother:
        ``supersmoother``. Gaussian regressions: ``ridge``, ``lasso``, ``elasticnet``.
    break_tolerance : float, default: window_length/2
        If there are large gaps in time (larger than ``window_length``/2), flatten will
        split the flux into several sub-lightcurves and apply the filter to each
        individually. ``break_tolerance`` must be in the same unit as ``time`` (usually
        days). To disable this feature, set ``break_tolerance`` to 0. If the method is
        ``supersmoother`` and no ``break_tolerance`` is provided, it will be taken as
        `1` in units of ``time``.
    edge_cutoff : float, default: None
        Trends near edges are less robust. Depending on the data, it may be beneficial
        to remove edges. The ``edge_cutoff`` defines the length (in units of time) to be
        cut off each edge. Default: Zero. Cut off is maximally ``window_length``/2, as
        this fills the window completely. Applicable only for time-windowed sliders.
    cval : float or int
        Tuning parameter for the robust estimators. See documentation for defaults. 
        Larger values for make the estimate more efficient but less robust. For the 
        super-smoother, cval determines the bass enhancement (smoothness) and can be 
        `None` or in the range 0 < ``cval`` < 10. For the ``savgol``, ``cval`` 
        determines the (integer) polynomial order (default: 2).
    proportiontocut : float, default: 0.1
        Fraction to cut off (or filled) of both tails of the distribution using methods
        ``trim_mean`` (or ``winsorize``)
    kernel : str, default: `squared_exp`
        Choice of `squared_exp` (squared exponential), `matern`, `periodic`,
        `periodic_auto`.
    kernel_size : float, default: 1
        The length scale of the Gaussian Process kernel.
    kernel_period : float
        The periodicity of the Gaussian Process kernel (in units of ``time``). Must be
        provided for the kernel `periodic`. Can not be specified for the
        `periodic_auto`, for which it is determined automatically using a Lomb-Scargle
        periodogram pre-search.
    robust : bool, default: False
        If `True`, the fitting process will be run iteratively. In each iteration,
        2-sigma outliers from the fitted trend will be  clipped until convergence.
        Supported by the Gaussian Process kernels `squared_exp` and `matern`, as well as
        `cosine` fitting.

    return_trend : bool, default: False
        If `True`, the method will return a tuple of two elements
        (``flattened_flux``, ``trend_flux``) where ``trend_flux`` is the removed trend.

    Returns
    -------
    flatten_flux : array-like
        Flattened flux.
    trend_flux : array-like
        Trend in the flux. Only returned if ``return_trend`` is `True`.
    """
    if method not in constants.methods:
        raise ValueError('Unknown detrending method')

    # Numba can't handle strings, so we're passing the location estimator as an int:
    if method == 'biweight':
        method_code = 1
    elif method == 'andrewsinewave':
        method_code = 2
    elif method == 'welsch':
        method_code = 3
    elif method == 'hodges':
        method_code = 4
    elif method == 'median':
        method_code = 5
    elif method == 'mean':
        method_code = 6
    elif method == 'trim_mean':
        method_code = 7
    elif method == 'winsorize':
        method_code = 8
    elif method == 'hampelfilt':
        method_code = 9
    elif method == 'huber_psi':
        method_code = 10
    elif method == 'tau':
        method_code = 11

    error_text = 'proportiontocut must be >0 and <0.5'
    if not isinstance(proportiontocut, float):
        raise ValueError(error_text)
    if proportiontocut >= 0.5 or proportiontocut <= 0:
        raise ValueError(error_text)

    # Default cval values for robust location estimators
    if cval is None:
        if method == 'biweight':
            cval = 5
        elif method == 'andrewsinewave':
            cval = 1.339
        elif method == 'welsch':
            cval = 2.11
        elif method == 'huber':
            cval = 1.5
        elif method == 'huber_psi':
            cval = 1.28
        elif method in ['trim_mean', 'winsorize']:
            cval = proportiontocut
        elif method == 'hampelfilt':
            cval = 3
        elif method == 'tau':
            cval = 4.5
        elif method == 'hampel':
            cval = (1.7, 3.4, 8.5)
        elif method == 'ramsay':
            cval = 0.3
        elif method == 'savgol':  # polyorder
            cval = 2  # int
        elif method in 'ridge lasso elasticnet':
            cval = 1
        else:
            cval = 0  # avoid numba type inference error: None type multi with float

    if cval is not None and method == 'supersmoother':
        if cval > 0 and cval < 10:
            supersmoother_alpha = cval
        else:
            supersmoother_alpha = None

    # Maximum gap in time should be half a window size.
    # Any larger is nonsense,  because then the array has a full window of data
    if window_length is None:
        window_length = 2  # so that break_tolerance = 1 in the supersmoother case
    if break_tolerance is None:
        break_tolerance = window_length / 2
    if break_tolerance == 0:
        break_tolerance = inf

    # Numba is very fast, but doesn't play nicely with NaN values
    # Therefore, we make new time-flux arrays with only the floating point values
    # All calculations are done within these arrays
    # Afterwards, the trend is transplanted into the original arrays (with the NaNs)
    if mask is None:
        mask = np.ones(len(time))
    else:
        mask = array(~mask, dtype=float64)  # Invert to stay consistent with TLS
    time = array(time, dtype=float64)
    flux = array(flux, dtype=float64)
    
    mask_nans = isnan(time * flux)
    time_compressed = np.ma.compressed(np.ma.masked_array(time, mask_nans))
    flux_compressed = np.ma.compressed(np.ma.masked_array(flux, mask_nans))
    mask_compressed = np.ma.compressed(np.ma.masked_array(mask, mask_nans))

    # Get the indexes of the gaps
    gaps_indexes = get_gaps_indexes(time_compressed, break_tolerance=break_tolerance)
    trend_flux = array([])
    trend_segment = array([])
    nsplines = array([])  # Chosen number of splines per segment for method "pspline"

    # Iterate over all segments
    for i in range(len(gaps_indexes) - 1):
        time_view = time_compressed[gaps_indexes[i]:gaps_indexes[i+1]]
        flux_view = flux_compressed[gaps_indexes[i]:gaps_indexes[i+1]]
        mask_view = mask_compressed[gaps_indexes[i]:gaps_indexes[i+1]]
        methods = ["biweight", "andrewsinewave", "welsch", "hodges", "median", "mean",
            "trim_mean", "winsorize", "huber_psi", "hampelfilt", "tau"]
        if method in methods:
            trend_segment = running_segment(
                time_view,
                flux_view,
                mask_view,
                window_length,
                edge_cutoff,
                cval,
                method_code)
        elif method in ["huber", "hampel", "ramsay"]:
            trend_segment = running_segment_slow(
                time_view,
                flux_view,
                mask_view,
                window_length,
                edge_cutoff,
                cval,
                method
                )
        elif method == 'lowess':
            trend_segment = lowess(
                time_view,
                flux_view,
                mask_view,
                window_length
                )
        elif method == 'hspline':
            trend_segment = detrend_huber_spline(
                time_view,
                flux_view,
                mask_view,
                knot_distance=window_length)
        elif method == 'supersmoother':
            try:
                from supersmoother import SuperSmoother as supersmoother
            except:
                raise ImportError('Could not import supersmoother')
            win = window_length / (max(time)-min(time))
            trend_segment = supersmoother(
                alpha=supersmoother_alpha,
                primary_spans=(
                    constants.primary_span_lower * win, 
                    win,
                    constants.primary_span_upper * win
                    ),
                middle_span=constants.middle_span * win,
                final_span=constants.upper_span * win
                ).fit(time_view, flux_view,).predict(time_view)
        elif method == 'cofiam':
            trend_segment = detrend_cofiam(
                time_view, flux_view, window_length)
        elif method == 'cosine':
            trend_segment = detrend_cosine(
                time_view, flux_view, window_length, robust, mask_view)
        elif method == 'savgol':
            if window_length%2 == 0:
                window_length += 1
            trend_segment = savgol_filter(flux_view, window_length, polyorder=int(cval))
        elif method == 'medfilt':
            trend_segment = medfilt(flux_view, window_length)
        elif method == 'gp':
            trend_segment = make_gp(
                time_view,
                flux_view,
                mask_view,
                kernel,
                kernel_size,
                kernel_period,
                robust
                )
        elif method == 'rspline':
            trend_segment = iter_spline(time_view, flux_view, mask_view, window_length)
        elif method == 'pspline':
            if verbose:
                print('Segment', i + 1, 'of', len(gaps_indexes) - 1)
            trend_segment, nsplines_segment = pspline(
                time_view, flux_view, edge_cutoff, max_splines, return_nsplines, verbose
                )
            nsplines = append(nsplines, nsplines_segment)
        elif method in "ridge lasso elasticnet":
            trend_segment = regression(time_view, flux_view, method, window_length, cval)

        trend_flux = append(trend_flux, trend_segment)            

    # Insert results of non-NaNs into original data stream
    trend_lc = full(len(time), nan)
    mask_nans = where(~mask_nans)[0]
    for idx in range(len(mask_nans)):
        trend_lc[mask_nans[idx]] = trend_flux[idx]
    trend_lc[trend_lc==0] = np.nan  # avoid division by zero
    flatten_lc = flux / trend_lc

    if return_trend and return_nsplines:
        return flatten_lc, trend_lc, nsplines
    if return_trend and not return_nsplines:
        return flatten_lc, trend_lc
    if not return_trend and not return_nsplines:
        return flatten_lc
