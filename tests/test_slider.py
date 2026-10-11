"""Tests of the time-windowed slider machinery (``wotan.slider`` via ``flatten``).

The window semantics asserted here are the *documented* ones: the trend at
time t_i is the location estimate of all fluxes with ``|t - t_i| < window/2``.
Each slider is checked against a brute-force implementation of exactly that
rule, using the *same* estimator function on the brute-force window, so these
tests isolate the windowing / masking / edge logic from the estimator maths.

Windows that reach the final sample are excluded from the reference checks
(the last sample used to be missing from every window, BUG-002; the check is kept
for windows away from the end and the end itself is covered in test_bug_regressions.py).
"""
import numpy as np
import pytest

from wotan import flatten

from helpers import away_from_end, interior, window_reference
from test_location_estimates import ESTIMATORS

# flatten method name -> estimator with that method's default cval
SLIDERS = [
    "biweight", "andrewsinewave", "welsch", "hodges", "median", "mean",
    "trim_mean", "winsorize", "hampelfilt", "huber_psi",
]
WINDOW = 1.0


def irregular_lc(n=400, span=10.0, seed=3, level=1.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n))
    f = level + rng.normal(0.0, 1e-2, n) + 0.02 * np.sin(t)
    return t, f


# --------------------------------------------------------- window semantics
@pytest.mark.parametrize("method", SLIDERS)
def test_slider_equals_bruteforce_window_irregular_sampling(method):
    t, f = irregular_lc()
    trend = flatten(t, f, window_length=WINDOW, method=method, break_tolerance=0,
                    return_trend=True)[1]
    ref = window_reference(t, f, WINDOW, ESTIMATORS[method])
    ok = away_from_end(t, WINDOW)
    np.testing.assert_allclose(trend[ok], ref[ok], rtol=0, atol=2e-6)


@pytest.mark.parametrize("method", SLIDERS)
def test_slider_equals_bruteforce_window_even_sampling(method):
    t = np.linspace(0, 10, 501)  # dt = 0.02
    f = 1 + 0.02 * np.sin(t) + np.random.default_rng(1).normal(0, 1e-2, len(t))
    window = 1.005  # = 25.25 dt: window borders never coincide with a sample
    trend = flatten(t, f, window_length=window, method=method, break_tolerance=0,
                    return_trend=True)[1]
    ref = window_reference(t, f, window, ESTIMATORS[method])
    ok = away_from_end(t, window)
    np.testing.assert_allclose(trend[ok], ref[ok], rtol=0, atol=2e-6)


@pytest.mark.parametrize("method", SLIDERS)
def test_linear_ramp_is_reproduced_in_the_interior(method):
    # With a window containing 2m+1 samples centred on t_i, every reflection
    # equivariant location estimator returns the centre value of a linear ramp.
    dt, m = 0.01, 12
    t = np.arange(0, 20, dt)
    f = 1.0 + 0.05 * t
    window = (2 * m + 0.5) * dt  # exactly 2m+1 samples, margins of dt/4
    trend = flatten(t, f, window_length=window, method=method, break_tolerance=0,
                    return_trend=True)[1]
    sel = interior(t, window + 2 * dt)
    np.testing.assert_allclose(trend[sel], f[sel], rtol=0, atol=3e-6)


@pytest.mark.parametrize("method", ["median", "mean", "biweight"])
def test_constant_flux_gives_constant_trend(method):
    t = np.linspace(0, 10, 300)
    f = np.full_like(t, 7.5)
    trend = flatten(t, f, window_length=1.0, method=method, return_trend=True)[1]
    ok = away_from_end(t, 1.0)
    np.testing.assert_allclose(trend[ok], 7.5, atol=1e-12)


@pytest.mark.parametrize("method", SLIDERS)
def test_window_is_translation_invariant_in_flux_and_time(method):
    t, f = irregular_lc()
    base = flatten(t, f, window_length=WINDOW, method=method, return_trend=True)[1]
    shifted_flux = flatten(t, f + 100.0, window_length=WINDOW, method=method,
                           return_trend=True)[1]
    shifted_time = flatten(t + 2457000.0, f, window_length=WINDOW, method=method,
                           return_trend=True)[1]
    np.testing.assert_allclose(shifted_flux, base + 100.0, atol=3e-6)
    np.testing.assert_allclose(shifted_time, base, atol=1e-9)  # BJD offset (issue #51)


def test_window_larger_than_data_gives_single_value_for_all_points():
    t = np.linspace(0, 1, 50)
    f = 1 + np.random.default_rng(0).normal(0, 1e-2, 50)
    trend = flatten(t, f, window_length=1e6, method="median", return_trend=True)[1]
    # Every window covers all samples -> one common value
    assert np.ptp(trend) == 0.0
    assert trend[0] == pytest.approx(np.median(f))


# -------------------------------------------------------------- edge cutoff
@pytest.mark.parametrize("method", ["median", "biweight", "mean"])
def test_edge_cutoff_blanks_exactly_the_edges(method):
    t = np.linspace(0, 20, 801)
    f = 1 + 0.01 * np.sin(t)
    cut = 0.7
    full = flatten(t, f, window_length=2.0, method=method, return_trend=True)[1]
    cutd = flatten(t, f, window_length=2.0, edge_cutoff=cut, method=method,
                   return_trend=True)[1]
    in_cut = (t < t.min() + cut) | (t > t.max() - cut)
    assert np.isnan(cutd[in_cut]).all()
    np.testing.assert_array_equal(cutd[~in_cut], full[~in_cut])


def test_edge_cutoff_is_capped_at_half_a_window():
    t = np.linspace(0, 20, 801)
    f = np.ones_like(t)
    big = flatten(t, f, window_length=2.0, edge_cutoff=9.0, method="median",
                  return_trend=True)[1]
    half = flatten(t, f, window_length=2.0, edge_cutoff=1.0, method="median",
                   return_trend=True)[1]
    np.testing.assert_array_equal(np.isnan(big), np.isnan(half))


def test_edge_cutoff_applies_to_every_segment():
    t = np.r_[np.linspace(0, 5, 200), np.linspace(8, 13, 200)]
    f = np.ones_like(t)
    trend = flatten(t, f, window_length=1.0, edge_cutoff=0.3, break_tolerance=0.5,
                    method="median", return_trend=True)[1]
    for lo, hi in [(0, 5), (8, 13)]:
        seg = (t >= lo) & (t <= hi)
        edge = seg & ((t < lo + 0.3) | (t > hi - 0.3))
        assert np.isnan(trend[edge]).all()
        assert np.isfinite(trend[seg & ~edge]).all()


# -------------------------------------------------------------------- masks
def _masked_lc():
    t, f = irregular_lc(n=500, seed=8)
    mask = np.zeros(len(t), bool)
    # narrower than the window (1.0), so no window is ever *entirely* masked
    mask[(t > 4.2) & (t < 4.8)] = True
    f = f.copy()
    f[mask] += 5.0  # absurd values under the mask: must be invisible
    return t, f, mask


@pytest.mark.parametrize("method", SLIDERS)
def test_masked_points_do_not_influence_the_trend(method):
    t, f, mask = _masked_lc()
    trend = flatten(t, f, window_length=WINDOW, method=method, mask=mask,
                    break_tolerance=0, return_trend=True)[1]
    keep = ~mask
    ref = window_reference(t, f, WINDOW, ESTIMATORS[method], mask_keep=keep)
    ok = away_from_end(t, WINDOW)
    np.testing.assert_allclose(trend[ok], ref[ok], rtol=0, atol=2e-6)
    assert np.isfinite(trend[mask & ok]).all()  # trend is still defined under the mask


def test_mask_follows_the_TLS_convention_true_means_excluded():
    t = np.linspace(0, 10, 400)
    f = np.ones_like(t)
    f[(t > 4.5) & (t < 5.5)] = 0.5
    mask = (t > 4.5) & (t < 5.5)
    kept = flatten(t, f, window_length=3.0, mask=mask, method="mean", return_trend=True)[1]
    inverted = flatten(t, f, window_length=3.0, mask=~mask, method="mean", return_trend=True)[1]
    assert np.nanmax(np.abs(kept - 1.0)) < 1e-9
    assert np.nanmax(np.abs(inverted - 1.0)) > 0.1


@pytest.mark.parametrize("method", ["median", "biweight", "mean"])
def test_everything_masked_gives_nan_not_a_crash(method):
    t = np.linspace(0, 10, 100)
    f = np.ones_like(t)
    trend = flatten(t, f, window_length=1.0, method=method, mask=np.ones(100, bool),
                    return_trend=True)[1]
    assert np.isnan(trend).all()


def test_mask_does_not_modify_user_arrays():
    t, f, mask = _masked_lc()
    t0, f0, m0 = t.copy(), f.copy(), mask.copy()
    flatten(t, f, window_length=WINDOW, method="biweight", mask=mask)
    np.testing.assert_array_equal(t, t0)
    np.testing.assert_array_equal(f, f0)
    np.testing.assert_array_equal(mask, m0)


def test_mask_with_nans_in_flux_is_aligned_correctly():
    t = np.linspace(0, 10, 400)
    f = np.ones_like(t) + 0.0
    f[50:60] = np.nan
    mask = np.zeros(400, bool)
    mask[200:240] = True
    f2 = f.copy()
    f2[200:240] = 9.0
    trend = flatten(t, f2, window_length=2.0, mask=mask, method="median",
                    return_trend=True)[1]
    ok = away_from_end(t, 2.0)
    assert np.isnan(trend[50:60]).all()
    np.testing.assert_allclose(trend[ok & ~np.isnan(trend)], 1.0, atol=1e-12)
