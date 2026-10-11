"""Behaviour of the public ``flatten`` interface, independent of the method used."""
import numpy as np
import pytest

import wotan.constants as constants
from wotan import flatten

from helpers import away_from_end, make_lc, sliding_mean_sine_bias

FAST = ["biweight", "median", "mean", "lowess", "cosine", "rspline"]


# --------------------------------------------------------- return contract
@pytest.mark.parametrize("method", FAST)
def test_flat_equals_flux_over_trend(method, lc):
    t, f, _ = lc
    flat, trend = flatten(t, f, window_length=3.0, method=method, return_trend=True)
    np.testing.assert_allclose(flat, f / trend, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("method", FAST)
def test_return_trend_flag_controls_the_return_type(method, lc):
    t, f, _ = lc
    only_flat = flatten(t, f, window_length=3.0, method=method)
    assert isinstance(only_flat, np.ndarray) and only_flat.shape == t.shape
    out = flatten(t, f, window_length=3.0, method=method, return_trend=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == out[1].shape == t.shape
    np.testing.assert_array_equal(out[0], only_flat)


def test_output_has_input_length_and_order_with_nans(lc):
    t, f, _ = lc
    f = f.copy()
    t = t.copy()
    f[100:110] = np.nan
    t[50] = np.nan
    flat, trend = flatten(t, f, window_length=2.0, return_trend=True)
    assert len(flat) == len(trend) == len(t)
    bad = np.zeros(len(t), bool)
    bad[100:110] = True
    bad[50] = True
    assert np.isnan(trend[bad]).all()
    assert np.isnan(flat[bad]).all()
    assert np.isfinite(trend[~bad]).all()


def test_nan_positions_do_not_change_values_elsewhere(lc):
    t, f, _ = lc
    base = flatten(t, f, window_length=2.0, return_trend=True)[1]
    f2 = f.copy()
    f2[700:710] = np.nan
    with_nan = flatten(t, f2, window_length=2.0, return_trend=True)[1]
    far = np.abs(t - t[705]) > 1.5
    np.testing.assert_array_equal(with_nan[far], base[far])


def test_zero_trend_becomes_nan_not_inf():
    t = np.linspace(0, 10, 100)
    flat, trend = flatten(t, np.zeros(100), window_length=1.0, return_trend=True)
    assert np.isnan(trend).all() and np.isnan(flat).all()  # documented in the source
    assert not np.isinf(flat).any()


def test_deterministic(lc):
    t, f, _ = lc
    a = flatten(t, f, window_length=2.0, method="biweight")
    b = flatten(t, f, window_length=2.0, method="biweight")
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------- input handling
def test_inputs_are_not_modified(lc):
    t, f, _ = lc
    t0, f0 = t.copy(), f.copy()
    for method in FAST + ["hspline", "hodges"]:
        flatten(t, f, window_length=3.0, method=method)
        np.testing.assert_array_equal(t, t0, err_msg=method)
        np.testing.assert_array_equal(f, f0, err_msg=method)


def test_accepts_lists_pandas_ints_and_float32(lc):
    t, f, _ = lc
    ref = flatten(t, f, window_length=2.0)
    np.testing.assert_allclose(flatten(list(t), list(f), window_length=2.0), ref)
    pd = pytest.importorskip("pandas")
    np.testing.assert_allclose(flatten(pd.Series(t), pd.Series(f), window_length=2.0), ref)
    assert np.isfinite(flatten(np.arange(300), np.ones(300) + 1e-3 * np.arange(300) % 3,
                               window_length=20)).all()
    out = flatten(t.astype(np.float32), f.astype(np.float32), window_length=2.0)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_default_method_is_biweight(lc):
    t, f, _ = lc
    np.testing.assert_array_equal(
        flatten(t, f, window_length=2.0),
        flatten(t, f, window_length=2.0, method="biweight"),
    )


def test_window_length_none_defaults_to_two_time_units(lc):
    t, f, _ = lc
    np.testing.assert_array_equal(
        flatten(t, f), flatten(t, f, window_length=2.0)
    )


# --------------------------------------------------------------- validation
def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown detrending method"):
        flatten(np.arange(10.0), np.ones(10), window_length=1.0, method="nonsense")


@pytest.mark.parametrize("p", [0.0, 0.5, 0.6, 1.0, -0.1])
def test_proportiontocut_out_of_range_raises(p):
    with pytest.raises(ValueError, match="proportiontocut"):
        flatten(np.arange(10.0), np.ones(10), window_length=1.0, method="trim_mean",
                proportiontocut=p)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        flatten(np.arange(10.0), np.ones(9), window_length=1.0)


def test_all_registered_methods_are_known_to_constants():
    documented = {
        "biweight", "lowess", "andrewsinewave", "welsch", "hodges", "median", "mean",
        "trim_mean", "hspline", "pspline", "cofiam", "supersmoother", "savgol", "medfilt",
        "gp", "rspline", "huber", "winsorize", "hampelfilt", "huber_psi", "hampel", "tau",
        "ramsay", "cosine", "ridge", "lasso", "elasticnet",
    }
    assert set(constants.methods) == documented


def test_documented_default_constants():
    assert constants.FTOL == 1e-6
    assert constants.PROPORTIONTOCUT == 0.1
    assert constants.PSPLINES_MAXITER == 10
    assert constants.PSPLINES_STDEV_CUT == 2
    assert constants.PSPLINES_MAX_SPLINES == 50


# ----------------------------------------------- documented cval defaults
@pytest.mark.parametrize(
    "method,cval",
    [
        ("biweight", 5), ("andrewsinewave", 1.339), ("welsch", 2.11),
        ("huber_psi", 1.28), ("hampelfilt", 3), ("tau", 4.5),
        ("trim_mean", 0.1), ("winsorize", 0.1),
    ],
)
def test_cval_defaults_match_documentation(method, cval):
    t, f, _ = make_lc(n=600, span=12.0)
    default = flatten(t, f, window_length=2.0, method=method)
    explicit = flatten(t, f, window_length=2.0, method=method, cval=cval)
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("method", ["biweight", "welsch", "andrewsinewave", "huber_psi"])
def test_cval_changes_the_result_when_outliers_are_present(method):
    t, f, _ = make_lc(n=600, span=12.0)
    f = f.copy()
    f[::37] += 0.01
    a = flatten(t, f, window_length=2.0, method=method, cval=1.0, return_trend=True)[1]
    b = flatten(t, f, window_length=2.0, method=method, cval=8.0, return_trend=True)[1]
    assert np.nanmax(np.abs(a - b)) > 1e-5


# ------------------------------------------------------ gaps / break_tolerance
def test_segments_are_detrended_independently():
    t = np.r_[np.linspace(0, 5, 200), np.linspace(5.8, 10, 168)]
    f = 1 + 0.002 * np.sin(t) + np.random.default_rng(0).normal(0, 1e-3, len(t))
    f2 = f.copy()
    f2[200:] *= 1.5  # flux offset after the gap
    whole = flatten(t, f2, window_length=1.0, break_tolerance=0.5, return_trend=True)[1]
    second = flatten(t[200:], f2[200:], window_length=1.0, break_tolerance=0.5,
                     return_trend=True)[1]
    first = flatten(t[:200], f2[:200], window_length=1.0, break_tolerance=0.5,
                    return_trend=True)[1]
    np.testing.assert_array_equal(whole[200:], second)  # exact: no leakage across the gap
    np.testing.assert_array_equal(whole[:200], first)
    assert np.nanmedian(whole[200:]) == pytest.approx(1.5, rel=0.02)


def test_default_break_tolerance_is_half_a_window():
    t = np.r_[np.linspace(0, 5, 200), np.linspace(5.6, 10, 176)]  # 0.6 gap > 0.5
    f = np.ones_like(t)
    f[200:] = 3.0
    default = flatten(t, f, window_length=1.0, return_trend=True)[1]
    explicit = flatten(t, f, window_length=1.0, break_tolerance=0.5, return_trend=True)[1]
    np.testing.assert_array_equal(default, explicit)
    assert default[199] == pytest.approx(1.0) and default[200] == pytest.approx(3.0)


def test_break_tolerance_zero_disables_splitting():
    t = np.r_[np.linspace(0, 5, 200), np.linspace(5.6, 10, 176)]
    f = np.ones_like(t)
    f[200:] = 3.0
    trend = flatten(t, f, window_length=2.0, break_tolerance=0, method="mean",
                    return_trend=True)[1]
    assert 1.0 < trend[199] < 3.0  # the window straddles the gap
    assert 1.0 < trend[200] < 3.0


def test_gap_smaller_than_break_tolerance_does_not_split():
    t = np.r_[np.linspace(0, 5, 200), np.linspace(5.3, 10, 188)]  # 0.3 gap < 0.5
    f = np.ones_like(t)
    f[200:] = 3.0
    trend = flatten(t, f, window_length=2.0, method="mean", return_trend=True)[1]
    assert 1.0 < trend[199] < 3.0


def test_segment_of_one_sample_gives_nan_but_does_not_crash():
    t = np.array([0.0, 0.01, 0.02, 5.0, 10.0, 10.01, 10.02])
    f = np.ones(7)
    trend = flatten(t, f, window_length=1.0, method="median", return_trend=True)[1]
    assert np.isnan(trend[3])  # isolated sample: no neighbours
    assert np.isfinite(trend[[0, 1, 4, 5]]).all()


# ------------------------------------------------------ light-curve physics
def test_flat_light_curve_is_flat(lc):
    t, f, true = lc
    window, noise, amp, period = 1.0, 5e-4, 0.01, 10.0
    flat = flatten(t, f, window_length=window, method="biweight")
    inner = (t > 2) & (t < 28)
    # residual scatter = white noise (+ the trend's own estimation noise, ~1/n_window)
    # plus the unavoidable sliding-window attenuation of the sinusoid
    n_w = window / (t[1] - t[0])
    bias = sliding_mean_sine_bias(amp, period, window)
    expected = np.sqrt(noise ** 2 * (1 + 2.0 / n_w) + bias ** 2 / 2)
    assert np.nanstd(flat[inner]) < 1.25 * expected
    assert np.nanmean(flat[inner]) == pytest.approx(1.0, abs=2e-4)


def test_flat_is_unchanged_when_flux_is_scaled(lc):
    t, f, _ = lc
    a = flatten(t, f, window_length=3.0, method="median")
    b = flatten(t, f * 28000.0, window_length=3.0, method="median")
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_memory_does_not_grow_with_repeated_calls():
    # issue #30 (numba leak); resource is POSIX-only
    resource = pytest.importorskip("resource")
    t = np.linspace(0, 1000, 20000)
    f = np.random.default_rng(0).normal(1, 1e-4, 20000)
    flatten(t, f, window_length=0.5)  # warm up / compile
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(25):
        flatten(t, f, window_length=0.5)
    growth_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r0) / 1024.0
    assert growth_mb < 50


@pytest.mark.parametrize("window", [0.5, 1.0, 2.0, 5.0])
def test_biweight_trend_stays_within_flux_range(window, lc):
    t, f, _ = lc
    trend = flatten(t, f, window_length=window, method="biweight", return_trend=True)[1]
    assert np.nanmin(trend) >= f.min() and np.nanmax(trend) <= f.max()


def test_trend_sampled_on_irregular_times_has_no_nans_in_dense_data():
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0, 20, 1000))
    f = 1 + rng.normal(0, 1e-3, 1000)
    trend = flatten(t, f, window_length=1.0, break_tolerance=0, return_trend=True)[1]
    assert np.isfinite(trend[away_from_end(t, 1.0)]).all()
