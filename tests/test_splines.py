"""Spline detrenders: ``rspline`` (iteratively sigma-clipped LSQ cubic spline),
``hspline`` (B-spline basis + Huber regression) and ``pspline`` (penalised GAM
spline, automatic knot density).

Definition-based facts used here:

* a least-squares cubic spline reproduces any cubic polynomial exactly
* knot spacing is ``window_length`` (rspline, hspline); pspline selects it itself
* masked points must not influence the fit
* all three are robust to short dips: a transit shorter than the knot spacing
  survives the flattening
"""
import numpy as np
import pytest

from wotan import flatten

from helpers import interior, make_lc, measured_depth, transit_scenario

NOISE = 5e-4


# --------------------------------------------------------------------- rspline
def test_rspline_reproduces_a_cubic_polynomial_exactly():
    t = np.linspace(0, 10, 300)
    cubic = 1 + 0.01 * t - 0.002 * t ** 2 + 0.0001 * t ** 3
    tr = flatten(t, cubic, window_length=2.0, method="rspline", return_trend=True)[1]
    np.testing.assert_allclose(tr, cubic, atol=1e-10)


def test_rspline_follows_the_trend(lc):
    t, f, true = lc
    tr = flatten(t, f, window_length=1.0, method="rspline", return_trend=True)[1]
    assert np.max(np.abs(tr - true)) < 6e-4  # measured 2.3e-4 with sigma 5e-4


def test_rspline_more_knots_fit_the_data_more_closely(lc):
    # In-sample residual scatter shrinks as the knot spacing shrinks (more parameters).
    t, f, _ = lc
    resid = {}
    for w in (0.3, 1.0, 5.0):
        tr = flatten(t, f, window_length=w, method="rspline", return_trend=True)[1]
        resid[w] = np.std(f - tr)
    assert resid[0.3] < resid[1.0] < resid[5.0]


def test_rspline_window_longer_than_span_gives_a_finite_cubic(lc):
    t, f, _ = lc
    tr = flatten(t, f, window_length=100.0, method="rspline", return_trend=True)[1]
    assert np.isfinite(tr).all()
    # no interior knots -> a single cubic polynomial: 4th differences vanish
    assert np.max(np.abs(np.diff(tr, 4))) < 1e-9


def test_rspline_clips_dips_without_needing_a_mask():
    t, f, true = make_lc(n=1500, span=30.0)
    f = f.copy()
    dip = (t > 14) & (t < 14.5)
    f[dip] -= 0.02
    tr = flatten(t, f, window_length=2.0, method="rspline", return_trend=True)[1]
    assert abs(np.mean(tr[dip] - true[dip])) < 0.05 * 0.02


def test_rspline_mask_is_equivalent_to_dropping_the_points():
    t, f, _ = make_lc(n=1500, span=30.0)
    drop = (t > 10) & (t < 12)
    f2 = f.copy()
    f2[drop] = 50.0  # absurd values the fit must never see
    f3 = f.copy()
    f3[drop] = np.nan
    a = flatten(t, f2, window_length=2.0, method="rspline", mask=drop, break_tolerance=0,
                return_trend=True)[1]
    b = flatten(t, f3, window_length=2.0, method="rspline", break_tolerance=0,
                return_trend=True)[1]
    np.testing.assert_allclose(a[~drop], b[~drop], atol=1e-9)


def test_rspline_too_few_points_gives_nan_not_an_exception():
    t = np.linspace(0, 1, 4)
    tr = flatten(t, np.ones(4), window_length=0.2, method="rspline", return_trend=True)[1]
    assert np.isnan(tr).all()


# --------------------------------------------------------------------- hspline
pytest.importorskip("sklearn")


def test_hspline_follows_the_trend(lc):
    t, f, true = lc
    tr = flatten(t, f, window_length=1.0, method="hspline", return_trend=True)[1]
    assert np.max(np.abs(tr - true)) < 6e-4  # measured 1.8e-4 with sigma 5e-4


def test_hspline_trend_is_not_forced_to_zero_issue_64():
    # Issue #64 reported trends "centred on 0". The spline itself tracks whatever
    # level the data have; it is the *division* in flatten() that explodes when the
    # flux crosses zero (flux must be positive, normalised ~1 - see BUGLOG N-1).
    # (An offset >> signal, e.g. +1000 on a 0.01 signal, loses precision: BUGLOG N-3.)
    t, f, true = make_lc(n=1500, span=30.0, level=1.0)
    for offset in (0.0, 5.0):
        tr = flatten(t, f - 1.0 + offset, window_length=1.0, method="hspline",
                     return_trend=True)[1]
        np.testing.assert_allclose(tr, true - 1.0 + offset, atol=6e-4)


@pytest.mark.parametrize("scale", [1e3, 28000.0])
def test_hspline_is_scale_equivariant_for_realistic_flux_levels(scale):
    t, f, _ = make_lc(n=1500, span=30.0)
    a = flatten(t, f, window_length=1.5, method="hspline", return_trend=True)[1]
    b = flatten(t, f * scale, window_length=1.5, method="hspline", return_trend=True)[1]
    np.testing.assert_allclose(b / scale, a, atol=1e-4)


def test_hspline_is_robust_to_a_short_dip_and_the_mask_helps_further():
    t, f, true = make_lc(n=1500, span=30.0)
    f = f.copy()
    dip = (t > 14) & (t < 14.6)
    f[dip] -= 0.02
    plain = flatten(t, f, window_length=2.0, method="hspline", return_trend=True)[1]
    masked = flatten(t, f, window_length=2.0, method="hspline", mask=dip,
                     return_trend=True)[1]
    assert abs(np.mean(plain[dip] - true[dip])) < 0.1 * 0.02
    assert abs(np.mean(masked[dip] - true[dip])) < abs(np.mean(plain[dip] - true[dip])) + 1e-4


def test_hspline_mask_is_equivalent_to_dropping_the_points():
    t, f, _ = make_lc(n=1500, span=30.0)
    drop = (t > 10) & (t < 12)
    f2 = f.copy()
    f2[drop] = 50.0
    f3 = f.copy()
    f3[drop] = np.nan
    a = flatten(t, f2, window_length=2.0, method="hspline", mask=drop, break_tolerance=0,
                return_trend=True)[1]
    b = flatten(t, f3, window_length=2.0, method="hspline", break_tolerance=0,
                return_trend=True)[1]
    np.testing.assert_allclose(a[~drop], b[~drop], atol=1e-6)


# ------------------------------------------------- depth preservation (paper 4.3)
@pytest.fixture(scope="module")
def scen():
    return transit_scenario(cadence_min=10.0)


@pytest.mark.parametrize("method", ["rspline", "hspline"])
def test_spline_knot_distance_3_t14_preserves_the_transit(scen, method):
    flat = flatten(scen["t"], scen["flux"], window_length=3 * scen["t14"], method=method)
    assert measured_depth(flat, scen["intransit"]) > 0.95 * scen["depth"]


def test_hspline_with_very_dense_knots_absorbs_the_transit(scen):
    flat = flatten(scen["t"], scen["flux"], window_length=0.5 * scen["t14"], method="hspline")
    assert measured_depth(flat, scen["intransit"]) < 0.3 * scen["depth"]


# --------------------------------------------------------------------- pspline
pygam = pytest.importorskip("pygam")


def test_pspline_follows_the_trend(lc):
    t, f, true = lc
    tr = flatten(t, f, window_length=3.0, method="pspline", return_trend=True)[1]
    assert np.max(np.abs(tr - true)) < 6e-4  # measured 2.4e-4


def test_pspline_selects_more_splines_for_more_wiggly_signals():
    t, f, _ = make_lc(n=1500, span=30.0)
    rng = np.random.default_rng(0)
    flat = 1 + rng.normal(0, NOISE, len(t))
    fast = 1 + 0.01 * np.sin(2 * np.pi * t / 2.0) + rng.normal(0, NOISE, len(t))
    n = {}
    for name, y in (("flat", flat), ("slow", f), ("fast", fast)):
        out = flatten(t, y, window_length=3.0, method="pspline", return_trend=True,
                      return_nsplines=True)
        n[name] = float(out[2][0])
    assert n["flat"] < n["slow"] <= n["fast"] <= 50


def test_pspline_return_nsplines_with_trend_gives_a_triple(lc):
    t, f, _ = lc
    out = flatten(t, f, window_length=3.0, method="pspline", return_trend=True,
                  return_nsplines=True)
    assert len(out) == 3
    assert out[2].shape == (1,) and out[2][0] >= 4


def test_pspline_max_splines_caps_the_flexibility():
    t, _, _ = make_lc(n=1500, span=30.0)
    fast = 1 + 0.01 * np.sin(2 * np.pi * t / 2.0)
    out = flatten(t, fast, window_length=3.0, method="pspline", max_splines=10,
                  return_trend=True, return_nsplines=True)
    assert out[2][0] <= 10 + 1  # edof of the penalised fit is bounded by the basis size


def test_pspline_edge_cutoff_blanks_exactly_the_edges(lc):
    t, f, _ = lc
    cut = 1.0
    tr = flatten(t, f, window_length=3.0, method="pspline", edge_cutoff=cut,
                 return_trend=True)[1]
    blank = (t <= t.min() + cut) | (t > t.max() - cut)
    assert np.isnan(tr[blank]).all()
    assert np.isfinite(tr[~blank]).all()


def test_pspline_ignores_a_narrow_dip():
    t, f, true = make_lc(n=1500, span=30.0)
    f = f.copy()
    dip = (t > 14) & (t < 14.3)
    f[dip] -= 0.01
    tr = flatten(t, f, window_length=3.0, method="pspline", return_trend=True)[1]
    assert abs(np.mean(tr[dip] - true[dip])) < 0.05 * 0.01


def test_pspline_gaps_smaller_than_the_break_tolerance_do_not_crash_issue_57():
    rng = np.random.default_rng(0)
    t = np.r_[np.linspace(0, 5, 150), np.linspace(5.9, 12, 180)]  # 0.9 gap < 1.5 tolerance
    f = 1 + 0.01 * np.sin(t) + rng.normal(0, NOISE, len(t))
    tr = flatten(t, f, window_length=3.0, method="pspline", return_trend=True)[1]
    assert np.isfinite(tr).all()
