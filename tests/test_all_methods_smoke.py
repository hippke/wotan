"""Every detrender must *run* and return a sensible trend (low-priority methods
are only smoke-tested; the priority ones have dedicated suites).

Accuracy bound: the sinusoid has amplitude 0.01, so a loose per-method bound
on |trend - truth| separates "works" from "garbage".
"""
import numpy as np
import pytest

import wotan.constants as constants
from wotan import flatten

from helpers import make_lc

# method -> (kwargs, max |trend - truth| allowed on the interior)
CASES = {
    "biweight": (dict(window_length=3.0), 8e-3),
    "andrewsinewave": (dict(window_length=3.0), 8e-3),
    "welsch": (dict(window_length=3.0), 8e-3),
    "hodges": (dict(window_length=3.0), 8e-3),
    "median": (dict(window_length=3.0), 8e-3),
    "mean": (dict(window_length=3.0), 8e-3),
    "trim_mean": (dict(window_length=3.0), 8e-3),
    "winsorize": (dict(window_length=3.0), 8e-3),
    "hampelfilt": (dict(window_length=3.0), 8e-3),
    "huber_psi": (dict(window_length=3.0), 8e-3),
    "tau": (dict(window_length=3.0), 8e-3),
    "huber": (dict(window_length=3.0), 8e-3),
    "hampel": (dict(window_length=3.0), 8e-3),
    "ramsay": (dict(window_length=3.0), 8e-3),
    "lowess": (dict(window_length=3.0), 2e-3),
    "cosine": (dict(window_length=3.0), 1e-3),
    "cofiam": (dict(window_length=3.0), 1e-3),
    "rspline": (dict(window_length=3.0), 2e-3),
    "hspline": (dict(window_length=1.5), 1e-3),
    "pspline": (dict(window_length=3.0), 1e-3),
    "supersmoother": (dict(window_length=3.0), 1.2e-2),
    "savgol": (dict(window_length=101), 1e-3),
    "medfilt": (dict(window_length=101), 1.5e-2),
    "gp_squared_exp": (dict(method="gp", kernel="squared_exp", kernel_size=3.0), 1e-3),
    "gp_matern": (dict(method="gp", kernel="matern", kernel_size=3.0), 1e-3),
    "gp_squared_exp_robust": (dict(method="gp", kernel="squared_exp", kernel_size=3.0,
                                   robust=True), 1e-3),
    "gp_periodic": (dict(method="gp", kernel="periodic", kernel_size=3.0,
                         kernel_period=10.0), 1e-3),
    "gp_periodic_auto": (dict(method="gp", kernel="periodic_auto", kernel_size=3.0), 1e-3),
    "cosine_robust": (dict(method="cosine", window_length=3.0, robust=True), 1e-3),
}

OPTIONAL = {
    "huber": "statsmodels", "hampel": "statsmodels", "ramsay": "statsmodels",
    "hspline": "sklearn", "pspline": "pygam", "supersmoother": "supersmoother",
}


@pytest.fixture(scope="module")
def data():
    return make_lc(n=1000, span=30.0)


@pytest.mark.parametrize("case", list(CASES))
def test_method_runs_and_tracks_the_trend(case, data):
    method = CASES[case][0].get("method", case)
    if method in OPTIONAL:
        pytest.importorskip(OPTIONAL[method])
    if method == "gp" or method == "hspline":
        pytest.importorskip("sklearn")
    t, f, true = data
    kwargs, bound = CASES[case]
    kwargs = dict(kwargs)
    kwargs.setdefault("method", case)
    flat, trend = flatten(t, f, return_trend=True, **kwargs)
    assert flat.shape == trend.shape == t.shape
    interior = (t > 3.0) & (t < 27.0)
    assert np.isfinite(trend[interior]).all(), "NaN/inf in the interior"
    assert np.max(np.abs(trend[interior] - true[interior])) < bound
    np.testing.assert_allclose(flat[interior], f[interior] / trend[interior], rtol=1e-12)


@pytest.mark.parametrize("method", ["ridge", "lasso", "elasticnet"])
@pytest.mark.xfail(
    reason="BUG-001: max_iter=1e5 is a float and is rejected by current scikit-learn "
    "(observed with 1.9.1); .travis.yml pins scikit-learn==0.20, which accepted it",
    strict=False,
)
def test_gaussian_basis_regressions_run(method, data):
    pytest.importorskip("sklearn")
    t, f, true = data
    trend = flatten(t, f, window_length=3.0, method=method, return_trend=True)[1]
    assert np.isfinite(trend).all()
    # (weakly flexible with default alpha: only check level and gross shape)
    assert np.max(np.abs(trend - true)) < 2e-2


@pytest.mark.parametrize("method", ["medfilt", "savgol"])
def test_cadence_based_methods_are_the_scipy_filters(method, data):
    from scipy.signal import medfilt, savgol_filter

    t, f, _ = data
    trend = flatten(t, f, window_length=51, method=method, return_trend=True)[1]
    ref = medfilt(f, 51) if method == "medfilt" else savgol_filter(f, 51, 2)
    np.testing.assert_allclose(trend, ref, atol=1e-12)


def test_savgol_makes_an_even_window_odd_and_cval_is_the_polyorder(data):
    from scipy.signal import savgol_filter

    t, f, _ = data
    even = flatten(t, f, window_length=50, method="savgol", return_trend=True)[1]
    np.testing.assert_allclose(even, savgol_filter(f, 51, 2), atol=1e-12)
    cubic = flatten(t, f, window_length=51, method="savgol", cval=3, return_trend=True)[1]
    np.testing.assert_allclose(cubic, savgol_filter(f, 51, 3), atol=1e-12)


def test_gp_requires_kernel_size_and_validates_it(data):
    pytest.importorskip("sklearn")
    t, f, _ = data
    with pytest.raises(ValueError, match="kernel_size must be specified"):
        flatten(t, f, method="gp", kernel="matern")
    for bad in (-1.0, 0.0, float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            flatten(t, f, method="gp", kernel="matern", kernel_size=bad)
    with pytest.raises(ValueError, match="floating point"):
        flatten(t, f, method="gp", kernel="matern", kernel_size="3")


def test_periodic_gp_requires_a_valid_period(data):
    pytest.importorskip("sklearn")
    t, f, _ = data
    with pytest.raises(ValueError, match="kernel_period must be specified"):
        flatten(t, f, method="gp", kernel="periodic", kernel_size=3.0)
    with pytest.raises(ValueError, match="finite and positive"):
        flatten(t, f, method="gp", kernel="periodic", kernel_size=3.0, kernel_period=-2.0)


def test_supersmoother_cval_controls_the_bass_enhancement(data):
    pytest.importorskip("supersmoother")
    t, f, _ = data
    a = flatten(t, f, window_length=3.0, method="supersmoother", cval=1, return_trend=True)[1]
    b = flatten(t, f, window_length=3.0, method="supersmoother", cval=9, return_trend=True)[1]
    assert np.max(np.abs(a - b)) > 1e-4


def test_robust_flag_only_changes_the_methods_that_support_it(data):
    t, f, _ = data
    # not supported by sliders: must be accepted and ignored
    a = flatten(t, f, window_length=3.0, method="median", return_trend=True)[1]
    b = flatten(t, f, window_length=3.0, method="median", robust=True, return_trend=True)[1]
    np.testing.assert_array_equal(a, b)
