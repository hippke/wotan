"""Wotan is a free and open source algorithm to automagically remove stellar trends
from light curves for exoplanet transit detection.
"""

from __future__ import print_function, division
import warnings
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


def _as_1d(values, name):
    """Convert input to a 1-d float array; accepts column vectors of shape (n, 1)."""
    values = np.asarray(values, dtype=float64)
    if values.ndim == 2 and 1 in values.shape:
        values = values.ravel()
    if values.ndim != 1:
        raise ValueError("%s must be one-dimensional" % name)
    return values


def flatten(
    time,
    flux,
    window_length=None,
    edge_cutoff=0,
    break_tolerance=None,
    cval=None,
    return_trend=False,
    method="biweight",
    kernel=None,
    kernel_size=None,
    kernel_period=None,
    proportiontocut=constants.PROPORTIONTOCUT,
    robust=False,
    max_splines=constants.PSPLINES_MAX_SPLINES,
    stdev_cut=constants.PSPLINES_STDEV_CUT,
    return_nsplines=False,
    mask=None,
    verbose=False,
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
        Detrending method. Time-windowed sliders: ``median``, ``biweight``, ``hodges``,
        ``tau``, ``welsch``, ``huber``, ``huber_psi``, ``andrewsinewave``, ``mean``,
        ``hampel``, ``ramsay``, ``trim_mean``, ``hampelfilt``, ``winsorize``. Cadence
        based slider: ``medfilt``. Splines: ``hspline``, ``rspline``, ``pspline``.
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
    edge_cutoff : float, default: 0
        Trends near edges are less robust. Depending on the data, it may be beneficial
        to remove edges. The ``edge_cutoff`` defines the length (in units of time) to be
        cut off each edge. Default: Zero. Cut off is maximally ``window_length``/2, as
        this fills the window completely. Applicable only for time-windowed sliders.
    cval : float or int
        Tuning parameter for the robust estimators. See documentation for defaults.
        Larger values for make the estimate more efficient but less robust. For the
        super-smoother, cval determines the bass enhancement (smoothness) and can be
        `None` or in the range 0 < ``cval`` < 10. For the ``savgol``, ``cval``
        determines the (integer) polynomial order (default: 2). For ``hspline``, ``cval``
        is the epsilon of the Huber regressor (default: 1.35). For ``lowess``, ``cval``
        is the cut-off of the bisquare robustness weights in units of the median
        absolute residual (default: 6, Cleveland 1979).
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
        `cosine` fitting. Outliers are clipped on both sides of the trend (dips and
        flares).
    max_splines : int, default: 100
        Maximum number of splines for the method ``pspline``. The optimum is searched
        between 1 and ``max_splines``.
    stdev_cut : float, default: 2
        Sigma threshold for the iterative outlier clipping of ``pspline``
        (``rspline`` and ``cosine`` with ``robust`` use the same default).
    return_nsplines : bool, default: False
        If `True`, the number of splines chosen for each segment by ``pspline`` is
        returned as an additional output (after ``trend_flux`` if ``return_trend`` is
        also `True`, otherwise directly after ``flatten_flux``).
    mask : array-like of bool, default: None
        Boolean array of the same length as ``time``. Points where ``mask`` is `True`
        are excluded from the trend fit (e.g. known transits); the trend is still
        evaluated at these points. Honoured by all methods except the cadence-based
        filters ``savgol`` and ``medfilt``, which emit a warning if a mask is given.
    verbose : bool, default: False
        If `True`, print progress information of the iterative methods.

    return_trend : bool, default: False
        If `True`, the method will return a tuple of two elements
        (``flattened_flux``, ``trend_flux``) where ``trend_flux`` is the removed trend.

    Returns
    -------
    flatten_flux : array-like
        Flattened flux.
    trend_flux : array-like
        Trend in the flux. Only returned if ``return_trend`` is `True`.

    Notes
    -----
    The flattened flux is ``flux / trend``. The flux therefore has to be positive and
    roughly normalised; a warning is issued if the trend is zero or negative. ``time``
    does not have to be sorted: the output is aligned with the input order. Non-finite
    values (NaN, inf) in ``time`` or ``flux`` are ignored and receive a NaN trend.
    """
    if method not in constants.methods:
        raise ValueError("Unknown detrending method")

    if window_length is not None:
        try:
            valid_window = np.isfinite(window_length) and window_length > 0
        except TypeError:
            valid_window = False
        if not valid_window:
            raise ValueError("window_length must be a positive, finite number")

    # Numba can't handle strings, so we're passing the location estimator as an int:
    if method == "biweight":
        method_code = 1
    elif method == "andrewsinewave":
        method_code = 2
    elif method == "welsch":
        method_code = 3
    elif method == "hodges":
        method_code = 4
    elif method == "median":
        method_code = 5
    elif method == "mean":
        method_code = 6
    elif method == "trim_mean":
        method_code = 7
    elif method == "winsorize":
        method_code = 8
    elif method == "hampelfilt":
        method_code = 9
    elif method == "huber_psi":
        method_code = 10
    elif method == "tau":
        method_code = 11

    error_text = "proportiontocut must be >0 and <0.5"
    if not isinstance(proportiontocut, float):
        raise ValueError(error_text)
    if proportiontocut >= 0.5 or proportiontocut <= 0:
        raise ValueError(error_text)

    # Default cval values for robust location estimators
    if cval is None:
        if method == "biweight":
            cval = 5
        elif method == "andrewsinewave":
            cval = 1.339
        elif method == "welsch":
            cval = 2.11
        elif method == "huber":
            cval = 1.5
        elif method == "huber_psi":
            cval = 1.28
        elif method in ["trim_mean", "winsorize"]:
            cval = proportiontocut
        elif method == "hampelfilt":
            cval = 3
        elif method == "tau":
            cval = 4.5
        elif method == "hampel":
            cval = (1.7, 3.4, 8.5)
        elif method == "ramsay":
            cval = 0.3
        elif method == "savgol":  # polyorder
            cval = 2  # int
        elif method == "hspline":  # epsilon of the Huber regressor
            cval = 1.35
        elif method == "lowess":  # bisquare cut-off in units of the median abs. residual
            cval = 6.0
        elif method in "ridge lasso elasticnet":
            cval = 1
        else:
            cval = 0  # avoid numba type inference error: None type multi with float

    if cval is not None and method == "supersmoother":
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
    time = _as_1d(time, "time")
    flux = _as_1d(flux, "flux")
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length")
    if len(time) == 0:
        raise ValueError("time and flux must not be empty")
    mask_given = mask is not None
    if mask is None:
        mask = np.ones(len(time))
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != time.shape:
            raise ValueError("mask must have the same length as time")
        mask = array(~mask, dtype=float64)  # Invert to stay consistent with TLS
    if mask_given and method in ("savgol", "medfilt"):
        warnings.warn(
            "The mask is ignored by the cadence-based filter '%s'" % method, UserWarning
        )

    # The sliders need ascending time. Sort internally, restore the order at the end
    order = None
    finite_time = time[np.isfinite(time)]
    if len(finite_time) > 1 and np.any(np.diff(finite_time) < 0):
        order = np.argsort(time, kind="stable")
        time, flux, mask = time[order], flux[order], mask[order]

    # Non-finite values (NaN or inf) in time or flux are invalid
    mask_nans = ~(np.isfinite(time) & np.isfinite(flux))
    if mask_nans.all():
        raise ValueError("time and flux contain no finite values")
    if mask_nans.any():
        valid = ~mask_nans
        time_compressed, flux_compressed, mask_compressed = (
            time[valid], flux[valid], mask[valid])
    else:  # nothing to remove: no copies
        time_compressed, flux_compressed, mask_compressed = time, flux, mask

    # Get the indexes of the gaps
    gaps_indexes = get_gaps_indexes(time_compressed, break_tolerance=break_tolerance)
    trend_parts = []  # trend of each segment, joined once at the end
    trend_segment = array([])
    nsplines = array([])  # Chosen number of splines per segment for method "pspline"

    # Iterate over all segments
    for i in range(len(gaps_indexes) - 1):
        time_view = time_compressed[gaps_indexes[i] : gaps_indexes[i + 1]]
        flux_view = flux_compressed[gaps_indexes[i] : gaps_indexes[i + 1]]
        mask_view = mask_compressed[gaps_indexes[i] : gaps_indexes[i + 1]]
        methods = [
            "biweight",
            "andrewsinewave",
            "welsch",
            "hodges",
            "median",
            "mean",
            "trim_mean",
            "winsorize",
            "huber_psi",
            "hampelfilt",
            "tau",
        ]
        if method in methods:
            trend_segment = running_segment(
                time_view,
                flux_view,
                mask_view,
                window_length,
                edge_cutoff,
                cval,
                method_code,
            )
        elif method in ["huber", "hampel", "ramsay"]:
            trend_segment = running_segment_slow(
                time_view,
                flux_view,
                mask_view,
                window_length,
                edge_cutoff,
                cval,
                method,
            )
        elif method == "lowess":
            trend_segment = lowess(
                time_view, flux_view, mask_view, window_length, float(cval)
            )
        elif method == "hspline":
            trend_segment = detrend_huber_spline(
                time_view, flux_view, mask_view, knot_distance=window_length, cval=cval
            )
        elif method == "supersmoother":
            try:
                from supersmoother import SuperSmoother as supersmoother
            except ImportError:
                raise ImportError("Could not import supersmoother")
            win = window_length / (max(time_view) - min(time_view))
            use = mask_view > 0  # fit on the unmasked points, predict everywhere
            trend_segment = (
                supersmoother(
                    alpha=supersmoother_alpha,
                    primary_spans=(
                        constants.primary_span_lower * win,
                        win,
                        constants.primary_span_upper * win,
                    ),
                    middle_span=constants.middle_span * win,
                    final_span=constants.upper_span * win,
                )
                .fit(
                    time_view[use],
                    flux_view[use],
                )
                .predict(time_view)
            )
        elif method == "cofiam":
            trend_segment = detrend_cofiam(time_view, flux_view, window_length, mask_view)
        elif method == "cosine":
            trend_segment = detrend_cosine(
                time_view, flux_view, window_length, robust, mask_view, verbose
            )
        elif method == "savgol":
            if window_length % 2 == 0:
                window_length += 1
            trend_segment = savgol_filter(flux_view, window_length, polyorder=int(cval))
        elif method == "medfilt":
            trend_segment = medfilt(flux_view, window_length)
        elif method == "gp":
            trend_segment = make_gp(
                time_view,
                flux_view,
                mask_view,
                kernel,
                kernel_size,
                kernel_period,
                robust,
                verbose,
            )
        elif method == "rspline":
            trend_segment = iter_spline(time_view, flux_view, mask_view, window_length)
        elif method == "pspline":
            if verbose:
                print("Segment", i + 1, "of", len(gaps_indexes) - 1)
            trend_segment, nsplines_segment = pspline(
                time_view,
                flux_view,
                edge_cutoff,
                max_splines,
                stdev_cut,
                return_nsplines,
                verbose,
                mask_view,
            )
            nsplines = append(nsplines, nsplines_segment)
        elif method in "ridge lasso elasticnet":
            trend_segment = regression(
                time_view, flux_view, method, window_length, cval, mask_view
            )

        trend_parts.append(np.asarray(trend_segment, dtype=float64).ravel())

    # Insert results of valid points into original data stream
    invalid_input = mask_nans
    trend_lc = full(len(time), nan)
    trend_lc[~mask_nans] = np.concatenate(trend_parts)
    if np.any(trend_lc <= 0):
        warnings.warn(
            "The trend is zero or negative at %d points. flatten divides the flux by "
            "the trend, so the flux must be positive and roughly normalised "
            "(e.g. divided by its median); the flattened flux is not meaningful here."
            % np.sum(trend_lc <= 0),
            UserWarning,
        )
    n_no_trend = int(np.sum(~invalid_input & isnan(trend_lc)))
    if n_no_trend > 0 and edge_cutoff == 0:
        warnings.warn(
            "%d points with valid flux received a NaN trend (their window contains "
            "no other usable point, e.g. an isolated sample, a fully masked window, "
            "or window_length too large for the method)." % n_no_trend,
            UserWarning,
        )
    trend_lc[trend_lc == 0] = np.nan  # avoid division by zero
    flatten_lc = flux / trend_lc

    if order is not None:  # restore the order of the input
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        flatten_lc, trend_lc = flatten_lc[inverse], trend_lc[inverse]

    if return_trend and return_nsplines:
        return flatten_lc, trend_lc, nsplines
    if return_trend:
        return flatten_lc, trend_lc
    if return_nsplines:
        return flatten_lc, nsplines
    return flatten_lc
