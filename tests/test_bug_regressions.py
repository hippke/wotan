"""Regression tests, one per BUGLOG.md entry (BUG-001 ... BUG-031).

Every test asserts the *correct* behaviour that was violated by the bug, as
defined by the method / the wotan paper / the documentation (cited in each
test).  They failed before the fixes (as strict ``xfail`` tests) and must keep
passing now.  ``bug(...)`` only documents the original problem.
"""
import importlib
import warnings

import numpy as np
import pytest

from wotan import flatten, slide_clip
from wotan.location_estimates import huber_psi, location_iter, tau, winsorize_mean
from wotan.slider import running_segment

from helpers import make_lc, window_reference


def bug(ident, why, strict=True):
    """Documentation marker: the original problem (the bug is fixed)."""
    return lambda func: func


# ---------------------------------------------------------------------- BUG-001
@pytest.mark.parametrize("method", ["ridge", "lasso", "elasticnet"])
@bug("BUG-001", "regression.py passes max_iter=1e5 (float) to scikit-learn", strict=False)
def test_bug001_ridge_lasso_elasticnet_run_with_current_sklearn(method):
    pytest.importorskip("sklearn")
    t, f, _ = make_lc(n=500, span=20.0)
    trend = flatten(t, f, window_length=3.0, method=method, return_trend=True)[1]
    assert np.isfinite(trend).all()


# ---------------------------------------------------------------------- BUG-002
@bug("BUG-002", "slider window upper bound is exclusive and capped at size-1: "
                "the last sample of a segment is never used")
def test_bug002_last_sample_is_part_of_the_window():
    t = np.array([0.0, 1.0, 2.0])
    f = np.array([1.0, 1.0, 100.0])
    trend = flatten(t, f, window_length=10.0, method="mean", break_tolerance=0,
                    return_trend=True)[1]
    np.testing.assert_allclose(trend, np.mean(f))  # all three windows hold all 3 points


@bug("BUG-002", "every window reaching the last sample is missing that sample")
def test_bug002_windows_near_the_end_match_the_bruteforce_definition():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 10, 400))
    f = 1 + rng.normal(0, 1e-2, 400)
    trend = flatten(t, f, window_length=1.0, method="median", break_tolerance=0,
                    return_trend=True)[1]
    ref = window_reference(t, f, 1.0, np.median)
    np.testing.assert_allclose(trend, ref, atol=1e-12)  # incl. the last ~window/2


@bug("BUG-002", "same off-by-one in the statsmodels-based sliders")
def test_bug002_huber_slider_includes_the_last_sample():
    sm_scale = pytest.importorskip("statsmodels.robust.scale")
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    f = np.array([1.0, 1.1, 0.9, 1.05, 1.6])
    trend = flatten(t, f, window_length=10.0, method="huber", break_tolerance=0,
                    return_trend=True)[1]
    ref = float(sm_scale.Huber(maxiter=1000, tol=1e-6, c=1.5)(f)[0])  # all 5 samples: 1.1135
    assert trend[0] == pytest.approx(ref, abs=1e-9)  # wotan gives 1.0125 (first 4 only)


# ---------------------------------------------------------------------- BUG-003
@bug("BUG-003", "running_segment hides masked points as negative flux and then drops "
                "every value <= 0, so non-positive flux is silently discarded")
def test_bug003_constant_negative_flux_gives_that_constant():
    t = np.linspace(0, 10, 100)
    trend = flatten(t, np.full(100, -5.0), window_length=2.0, method="median",
                    break_tolerance=0, return_trend=True)[1]
    ok = t + 1.0 < t.max()
    np.testing.assert_allclose(trend[ok], -5.0)


@bug("BUG-003", "zero-mean (mean-subtracted) flux loses half of its samples")
def test_bug003_zero_mean_flux_has_a_complete_trend():
    t = np.linspace(0, 10, 400)
    f = 0.01 * np.sin(t)  # symmetric around 0
    trend = flatten(t, f, window_length=1.0, method="median", break_tolerance=0,
                    return_trend=True)[1]
    assert np.isnan(trend[t + 0.5 < t.max()]).sum() == 0


@bug("BUG-003", "masked values of a negative-flux light curve flip sign and are INCLUDED")
def test_bug003_masked_negative_flux_does_not_leak():
    t = np.arange(20.0)
    f = np.full(20, -5.0)
    f[8:12] = -100.0
    mask = np.zeros(20, bool)
    mask[8:12] = True
    trend = flatten(t, f, window_length=6.0, method="median", mask=mask,
                    break_tolerance=0, return_trend=True)[1]
    np.testing.assert_allclose(trend[8:12], -5.0)


def test_bug003_control_positive_flux_masking_works():
    t = np.arange(20.0)
    f = np.full(20, 5.0)
    f[8:12] = 100.0
    mask = np.zeros(20, bool)
    mask[8:12] = True
    out = running_segment(t, f, np.where(mask, 0.0, 1.0), 6.0, 0.0, 0.0, 5)
    np.testing.assert_allclose(out[8:12], 5.0)


# ---------------------------------------------------------------------- BUG-004
@bug("BUG-004", "tau() divides by s = 0.675*MAD without a MAD == 0 guard "
                "(biweight/huber_psi/hampelfilt do have one)")
def test_bug004_tau_with_zero_mad_does_not_crash():
    x = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 100.0])  # MAD = 0
    v = tau(x, 4.5)
    assert np.isfinite(v)


@bug("BUG-004", "flatten(method='tau') crashes on flux that is constant over a window")
def test_bug004_tau_slider_on_flux_with_flat_stretches():
    t = np.linspace(0, 10, 300)
    f = np.ones(300)
    f[150] = 1.5
    trend = flatten(t, f, window_length=1.0, method="tau", return_trend=True)[1]
    assert np.isfinite(trend).all()


# ---------------------------------------------------------------------- BUG-005
@pytest.mark.parametrize("n", [2, 3, 5, 10, 11, 20, 50, 101])
@bug("BUG-005", "winsorize_mean winsorizes int(p*n)+1 values and treats the two tails "
                "differently; disagrees with scipy.stats.mstats.winsorize")
def test_bug005_winsorize_matches_scipy(n):
    from scipy.stats.mstats import winsorize

    x = np.random.default_rng(1).normal(0, 1, n)
    ref = np.mean(winsorize(x, limits=(0.1, 0.1)))
    assert winsorize_mean(x.copy(), 0.1) == pytest.approx(ref, abs=1e-12)


@bug("BUG-005", "upper tail is filled with a value from inside the replaced block")
def test_bug005_winsorize_is_reflection_antisymmetric():
    x = np.random.default_rng(3).normal(0, 1, 101) ** 3
    assert winsorize_mean(-x, 0.1) == pytest.approx(-winsorize_mean(x, 0.1), abs=1e-12)


@bug("BUG-005", "single-sample window reads sorted_data[1] of a 1-element array")
def test_bug005_winsorize_of_one_sample_is_that_sample():
    assert winsorize_mean(np.array([3.5]), 0.1) == 3.5


# ---------------------------------------------------------------------- BUG-006
@bug("BUG-006", "slide_clip documents method='mad'|'std', center='median'|'mean' but the "
                "code tests for 'median'|'mad' -> the defaults silently use std and mean")
def test_bug006_slide_clip_default_is_mad_around_the_median():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 400)
    f = 1 + rng.normal(0, 1e-3, 400)
    f[::20] += rng.choice([0.004, 0.02], 20)  # heavy contamination
    window = 1e3  # one window = everything
    out = slide_clip(t, f, window_length=window, low=3, high=3)  # defaults: mad / median
    med = np.median(f)
    mad = np.median(np.abs(f - med))
    expected_clipped = np.abs(f - med) > 3 * mad
    inner = slice(1, -2)  # slide_clip does not touch the first/last samples reliably
    np.testing.assert_array_equal(np.isnan(out)[inner], expected_clipped[inner])


@bug("BUG-006", "method='std' and method='mad' give identical results")
def test_bug006_mad_and_std_are_different_methods():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 400)
    f = 1 + rng.normal(0, 1e-3, 400)
    f[::20] += 0.01
    a = slide_clip(t, f, 1.0, method="mad", center="median")
    b = slide_clip(t, f, 1.0, method="std", center="mean")
    assert not np.array_equal(a, b, equal_nan=True)


# ---------------------------------------------------------------------- BUG-007
@bug("BUG-007", "slide_clip never writes the last sample (loop to size-1, exclusive "
                "window end), so the last point of every light curve comes back NaN")
def test_bug007_slide_clip_keeps_the_last_point_of_clean_data():
    t = np.linspace(0, 10, 200)
    f = np.ones(200)
    out = slide_clip(t, f, window_length=1.0)
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------- BUG-008
@bug("BUG-008", "kernel=None is the documented default but `'periodic' in kernel` "
                "raises TypeError")
def test_bug008_gp_default_kernel_is_squared_exponential():
    pytest.importorskip("sklearn")
    t, f, _ = make_lc(n=300, span=10.0)
    a = flatten(t, f, method="gp", kernel_size=3.0, return_trend=True)[1]
    b = flatten(t, f, method="gp", kernel="squared_exp", kernel_size=3.0, return_trend=True)[1]
    np.testing.assert_allclose(a, b)


@bug("BUG-008", "unknown kernel leaves trend_segment unbound (UnboundLocalError)")
def test_bug008_gp_unknown_kernel_is_a_clear_valueerror():
    pytest.importorskip("sklearn")
    t, f, _ = make_lc(n=300, span=10.0)
    with pytest.raises(ValueError):
        flatten(t, f, method="gp", kernel="not_a_kernel", kernel_size=3.0)


# ---------------------------------------------------------------------- BUG-009
@bug("BUG-009", "FTOL = 1e-6 is an absolute tolerance: the biweight is not scale "
                "equivariant, at flux ~1e-3 it degenerates to a one-step estimate")
def test_bug009_biweight_is_scale_equivariant_for_small_flux_units():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.01, 200)
    x[:5] += 0.2
    ref = location_iter(x.copy(), 5.0, 1)
    for s in (1e-3, 1e-6):
        assert location_iter(x * s, 5.0, 1) / s == pytest.approx(ref, rel=1e-6)


@bug("BUG-009", "lowess stops iterating on an absolute change < 1e-6")
def test_bug009_lowess_is_scale_equivariant_for_small_flux_units():
    t, f, _ = make_lc(n=1500, span=30.0)
    a = flatten(t, f, window_length=1.5, method="lowess", return_trend=True)[1]
    b = flatten(t, f * 1e-6, window_length=1.5, method="lowess", return_trend=True)[1] / 1e-6
    np.testing.assert_allclose(b, a, atol=1e-6)


# ---------------------------------------------------------------------- BUG-010
@bug("BUG-010", "detrend_cofiam loops k_m = 1..degree but always builds the basis with "
                "`degree`, so the Durbin-Watson selection (Kipping+2013) is a no-op")
def test_bug010_cofiam_tries_increasing_numbers_of_harmonics(monkeypatch):
    import wotan.cofiam as cof

    seen = []
    orig = cof.matrix_gen

    def spy(t, degree):
        seen.append(degree)
        return orig(t, degree)

    monkeypatch.setattr(cof, "matrix_gen", spy)
    t, f, _ = make_lc(n=400, span=20.0)
    cof.detrend_cofiam(t, f, 2.0)
    assert len(set(seen)) > 1


# ---------------------------------------------------------------------- BUG-011
@bug("BUG-011", "pspline iterates `range(stdev_cut)`: the sigma threshold doubles as "
                "the iteration count (float -> TypeError)")
def test_bug011_pspline_accepts_a_fractional_sigma_cut():
    pytest.importorskip("pygam")
    t, f, _ = make_lc(n=400, span=12.0)
    tr = flatten(t, f, window_length=1.0, method="pspline", stdev_cut=2.5,
                 return_trend=True)[1]
    assert np.isfinite(tr).all()


# ---------------------------------------------------------------------- BUG-012
@bug("BUG-012", "flatten has no branch for return_nsplines=True, return_trend=False "
                "-> returns None")
def test_bug012_return_nsplines_without_return_trend_returns_something():
    pytest.importorskip("pygam")
    t, f, _ = make_lc(n=300, span=10.0)
    out = flatten(t, f, window_length=1.0, method="pspline", return_nsplines=True)
    assert out is not None


# ---------------------------------------------------------------------- BUG-013
@bug("BUG-013", "unsorted time is not detected: sliders silently return wrong values and NaNs")
def test_bug013_unsorted_input_is_sorted_or_rejected():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 10, 300)
    f = 1 + 0.01 * np.sin(t) + rng.normal(0, 1e-3, 300)
    idx = rng.permutation(300)
    ref = flatten(t, f, window_length=1.0, return_trend=True)[1]
    try:
        out = flatten(t[idx], f[idx], window_length=1.0, return_trend=True)[1]
    except ValueError:
        return  # a loud failure is acceptable
    np.testing.assert_allclose(out[np.argsort(idx)], ref, atol=1e-12)


# ---------------------------------------------------------------------- BUG-014
@bug("BUG-014", "tau(): weights are only cut off when (1-(y/c)^2)^2 > c, so points "
                "with 1 < |y/c| < ~1.8 get weights up to 4.5; scale is 0.675*MAD (should "
                "divide by 0.6745)")
def test_bug014_tau_outliers_beyond_the_cutoff_get_no_weight():
    rng = np.random.default_rng(5)
    clean = rng.normal(0, 1, 101)
    contaminated = np.concatenate([clean, np.full(8, 3.3)])
    assert abs(tau(contaminated, 4.5) - tau(clean, 4.5)) < 0.3


# ---------------------------------------------------------------------- BUG-015
@bug("BUG-015", "huber_psi() counts inliers *after* clipping (always n), but the one-step "
                "Newton update divides by the number of unclipped points")
def test_bug015_huber_psi_is_the_newton_one_step():
    rng = np.random.default_rng(6)
    x = np.concatenate([rng.normal(0, 1, 100), np.full(20, 10.0)])
    c = 1.28
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    y = (x - med) / mad
    n_inliers = np.sum(np.abs(y) <= c)
    ref = med + mad * np.sum(np.clip(y, -c, c)) / n_inliers
    assert huber_psi(x.copy(), c) == pytest.approx(ref, abs=1e-9)


# ---------------------------------------------------------------------- BUG-016
@bug("BUG-016", "supersmoother span fraction uses max(time)-min(time) of the whole light "
                "curve instead of the segment: segments are not independent")
def test_bug016_supersmoother_segments_are_independent():
    pytest.importorskip("supersmoother")
    rng = np.random.default_rng(9)
    t = np.r_[np.linspace(0, 10, 300), np.linspace(15, 20, 150)]
    f = 1 + 0.01 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 5e-4, 450)
    whole = flatten(t, f, method="supersmoother", window_length=2.0, break_tolerance=1.0,
                    return_trend=True)[1]
    alone = flatten(t[300:], f[300:], method="supersmoother", window_length=2.0,
                    break_tolerance=1.0, return_trend=True)[1]
    np.testing.assert_allclose(whole[300:], alone, atol=1e-9)


# ---------------------------------------------------------------------- BUG-017
@bug("BUG-017", "hspline returns an all-NaN trend with no warning when fewer than 4 "
                "knots fit (window_length > span/3)")
def test_bug017_hspline_with_too_large_a_window_is_not_silently_nan():
    pytest.importorskip("sklearn")
    t, f, _ = make_lc(n=300, span=10.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            tr = flatten(t, f, window_length=4.0, method="hspline", return_trend=True)[1]
        except ValueError:
            return
    assert (not np.isnan(tr).all()) or any(
        issubclass(w.category, UserWarning) for w in caught
    )


# ---------------------------------------------------------------------- BUG-018
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
@bug("BUG-018", "window_length <= 0 or NaN is not validated: all-NaN output, no error")
def test_bug018_nonsensical_window_length_is_rejected(bad):
    t, f, _ = make_lc(n=200, span=10.0)
    with pytest.raises(ValueError):
        flatten(t, f, window_length=bad, method="biweight")


@bug("BUG-018", "flux of shape (n, 1) (e.g. DataFrame.to_numpy()) -> cryptic MaskError "
                "from a (n, n) broadcast (issue #54)")
def test_bug018_column_vector_flux_is_squeezed_or_rejected_clearly():
    t, f, _ = make_lc(n=300, span=10.0)
    try:
        out = flatten(t, f[:, None], window_length=1.0, method="median")
    except ValueError:
        return
    np.testing.assert_allclose(out, flatten(t, f, window_length=1.0, method="median"))


@bug("BUG-018", "mask given as a Python list -> TypeError from `~mask`")
def test_bug018_mask_may_be_a_list():
    t, f, _ = make_lc(n=300, span=10.0)
    m = np.zeros(300, bool)
    m[100:120] = True
    a = flatten(t, f, window_length=1.0, mask=list(m))
    b = flatten(t, f, window_length=1.0, mask=m)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------- BUG-019
@bug("BUG-019", "detrend_cosine prints 'Converged.' on every call, even when not robust "
                "(robust cosine / GP print per-iteration progress unconditionally)")
def test_bug019_library_is_silent_by_default(capsys):
    t, f, _ = make_lc(n=300, span=10.0)
    flatten(t, f, window_length=3.0, method="cosine")
    flatten(t, f, window_length=3.0, method="cosine", robust=True)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------- BUG-020
@bug("BUG-020", "iter_spline convergence test is dead code (np.ma.where on a tuple, "
                "second `outliers` never used): always runs PSPLINES_MAXITER fits")
def test_bug020_rspline_stops_when_no_more_points_are_clipped(monkeypatch):
    import wotan.constants as constants
    import wotan.iter_spline as its

    calls = []
    orig = its.LSQUnivariateSpline
    monkeypatch.setattr(its, "LSQUnivariateSpline",
                        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    # flat flux + white noise: the clipped set stops changing after ~5 fits
    # (a sinusoidal trend keeps shedding points, so it never converges in 10)
    t = np.linspace(0, 20, 600)
    f = 1 + np.random.default_rng(1).normal(0, 1e-4, 600)
    flatten(t, f, window_length=3.0, method="rspline")
    assert len(calls) < constants.PSPLINES_MAXITER


# ---------------------------------------------------------------------- BUG-021
@bug("BUG-021", "issue #55: flatten's docstring omits mask, max_splines, stdev_cut, "
                "return_nsplines and verbose")
def test_bug021_every_flatten_parameter_is_documented():
    import inspect

    doc = flatten.__doc__
    missing = [p for p in inspect.signature(flatten).parameters if p + " :" not in doc]
    assert not missing, missing


# ---------------------------------------------------------------------- BUG-022
@bug("BUG-022", "lowess tricube radius uses x[-2] (second to last window point), so the "
                "last point of a window at the left edge gets a non-zero (negative-cube) weight")
def test_bug022_lowess_edge_point_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.nonparametric.smoothers_lowess")
    t = np.linspace(0, 10, 200)
    k = 10
    f = np.ones(200)
    f[k - 1] += 1.0  # spike on the *last* member of the first window (zero tricube weight)
    ours = flatten(t, f, window_length=0.5, method="lowess", return_trend=True)[1]
    ref = sm.lowess(f, t, frac=k / 200.0, it=0, return_sorted=False)
    assert ours[0] == pytest.approx(ref[0], abs=1e-3)


# ---------------------------------------------------------------------- BUG-023
@bug("BUG-023", "wotan/norms.py runs demo code at import time and uses np.alltrue "
                "(removed in NumPy 2.0)")
def test_bug023_norms_module_can_be_imported():
    importlib.import_module("wotan.norms")


# ---------------------------------------------------------------------- BUG-024
@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("pspline", dict(window_length=3.0)),
        ("cofiam", dict(window_length=4.0)),
        ("supersmoother", dict(window_length=3.0)),
        ("savgol", dict(window_length=101)),
        ("medfilt", dict(window_length=101)),
    ],
)
@bug("BUG-024", "`mask` is silently ignored by pspline, cofiam, supersmoother, savgol, "
                "medfilt (and ridge/lasso/elasticnet): no effect, no warning")
def test_bug024_mask_is_honoured_or_the_user_is_told(method, kwargs):
    if method == "pspline":
        pytest.importorskip("pygam")
    if method == "supersmoother":
        pytest.importorskip("supersmoother")
    t, f, _ = make_lc(n=1000, span=30.0)
    f = f.copy()
    dip = (t > 14) & (t < 16)
    f[dip] -= 0.02
    plain = flatten(t, f, method=method, return_trend=True, **kwargs)[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            masked = flatten(t, f, method=method, mask=dip, return_trend=True, **kwargs)[1]
        except (ValueError, NotImplementedError):
            return  # refusing loudly is acceptable
    changed = not np.array_equal(plain, masked, equal_nan=True)
    warned = any("mask" in str(w.message).lower() for w in caught)
    assert changed or warned


# ---------------------------------------------------------------------- BUG-025
@bug("BUG-025", "flatten divides by the trend: zero-crossing flux gives ratios of ~100 "
                "(and, via BUG-003, silently dropped samples) with no warning")
def test_bug025_zero_crossing_flux_warns_or_is_rejected():
    t = np.linspace(0, 10, 300)
    f = 0.01 * np.sin(t)  # e.g. mean-subtracted flux (issue #64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = flatten(t, f, window_length=1.0, method="biweight")
        except ValueError:
            return
    assert any(issubclass(w.category, UserWarning) for w in caught) or np.nanmax(np.abs(out)) < 10


# ---------------------------------------------------------------------- BUG-026
@bug("BUG-026", "cosine builds its basis from raw t (columns 1, t, sin, cos): BJD-sized time "
                "offsets change the trend (ill-conditioning)")
def test_bug026_cosine_is_invariant_to_a_time_offset():
    t, f, _ = make_lc(n=1500, span=30.0)
    a = flatten(t, f, window_length=3.0, method="cosine", return_trend=True)[1]
    b = flatten(t + 2457000.0, f, window_length=3.0, method="cosine", return_trend=True)[1]
    np.testing.assert_allclose(b, a, atol=1e-6)


# ---------------------------------------------------------------------- BUG-027
@bug("BUG-027", "hspline's Huber regressor uses an absolute tolerance: a +1000 offset on a "
                "0.01-amplitude signal changes the trend by 4.5e-4 (biweight: exactly 0)")
def test_bug027_hspline_is_invariant_to_a_flux_offset():
    pytest.importorskip("sklearn")
    t, f, _ = make_lc(n=1500, span=30.0)
    a = flatten(t, f, window_length=3.0, method="hspline", return_trend=True)[1]
    b = flatten(t, f + 1000.0, window_length=3.0, method="hspline", return_trend=True)[1] - 1000.0
    np.testing.assert_allclose(b, a, atol=1e-5)


# ---------------------------------------------------------------------- BUG-028
@pytest.mark.parametrize("method,kwargs", [("cosine", dict(window_length=4.0, robust=True)),
                                           ("rspline", dict(window_length=4.0))])
@bug("BUG-028", "robust clipping is one-sided: a dip is ignored, an equally large flare "
                "pulls the trend (docstring: '2-sigma outliers')")
def test_bug028_flares_are_clipped_as_well_as_dips(method, kwargs):
    t, f, _ = make_lc(n=1000, span=20.0, noise=5e-4)
    bump = (t > 9.9) & (t < 10.3)
    up, down = f.copy(), f.copy()
    up[bump] += 0.03
    down[bump] -= 0.03
    base = flatten(t, f, method=method, return_trend=True, **kwargs)[1]
    eff_up = np.nanmax(np.abs(flatten(t, up, method=method, return_trend=True, **kwargs)[1] - base))
    eff_down = np.nanmax(np.abs(flatten(t, down, method=method, return_trend=True, **kwargs)[1] - base))
    assert eff_up < 2 * eff_down + 1e-4


# ---------------------------------------------------------------------- BUG-029
@bug("BUG-029", "an isolated one-sample segment gets a NaN trend with no warning "
                "(confidence: low, intended behaviour is not documented)")
def test_bug029_single_sample_segment_is_not_silently_nan():
    t = np.array([0, 0.01, 0.02, 5.0, 10, 10.01, 10.02])
    f = np.ones(7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tr = flatten(t, f, window_length=0.5, break_tolerance=1.0, method="median",
                     return_trend=True)[1]
    assert np.isfinite(tr).all() or any(issubclass(w.category, UserWarning) for w in caught)


# ---------------------------------------------------------------------- BUG-030
@bug("BUG-030", "bare `except:` clauses also swallow KeyboardInterrupt / SystemExit")
def test_bug030_no_bare_except_in_the_package():
    import ast
    import pathlib

    import wotan

    offenders = []
    for path in pathlib.Path(wotan.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                offenders.append("%s:%d" % (path.name, node.lineno))
    assert not offenders, offenders


# ---------------------------------------------------------------------- BUG-031
@bug("BUG-031", "only NaN is removed as invalid: +-inf in the flux corrupts neighbouring "
                "biweight trend values")
def test_bug031_inf_in_flux_is_treated_like_nan():
    t = np.linspace(0, 10, 300)
    f = 1 + 0.001 * np.sin(t)
    with_inf, with_nan = f.copy(), f.copy()
    with_inf[100] = np.inf
    with_nan[100] = np.nan
    a = flatten(t, with_inf, window_length=1.0, method="biweight", return_trend=True)[1]
    b = flatten(t, with_nan, window_length=1.0, method="biweight", return_trend=True)[1]
    np.testing.assert_allclose(a, b, equal_nan=True)
