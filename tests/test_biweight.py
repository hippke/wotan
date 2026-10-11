"""Tukey's biweight sliding location estimator (wotan's default and flagship method).

References: Tukey (1977); Mosteller & Tukey (1977); Hippke et al. (2019, AJ 158, 143),
Sect. 2 and the transit-injection experiments (best window ~ 3 x T14, cval = 5).
"""
import numpy as np
import pytest

from wotan import flatten

from helpers import (
    away_from_end,
    interior,
    make_lc,
    measured_depth,
    sliding_mean_sine_bias,
    transit_scenario,
)

NOISE, AMP, PERIOD = 5e-4, 0.01, 10.0


def test_follows_a_smooth_trend_within_theoretical_bias():
    t, f, true = make_lc(n=3000, span=40.0)
    w = 1.0
    trend = flatten(t, f, window_length=w, method="biweight", return_trend=True)[1]
    sel = interior(t, w)
    n_w = w / (t[1] - t[0])
    tol = 1.5 * sliding_mean_sine_bias(AMP, PERIOD, w) + 5 * NOISE / np.sqrt(n_w)
    assert np.max(np.abs(trend[sel] - true[sel])) < tol


@pytest.mark.parametrize("window", [0.5, 1.0, 2.0])
def test_trend_error_shrinks_with_the_window_bias_formula(window):
    # error budget = sliding attenuation + averaging noise; both are predictable
    t, f, true = make_lc(n=3000, span=40.0)
    trend = flatten(t, f, window_length=window, method="biweight", return_trend=True)[1]
    sel = interior(t, window)
    n_w = window / (t[1] - t[0])
    budget = 1.6 * sliding_mean_sine_bias(AMP, PERIOD, window) + 5 * NOISE / np.sqrt(n_w)
    assert np.max(np.abs(trend[sel] - true[sel])) < budget


def test_ignores_isolated_flares():
    t, f, true = make_lc(n=3000, span=40.0)
    f = f.copy()
    flare = np.arange(100, 3000, 150)
    f[flare] += 0.05  # 100 sigma single-cadence flares
    w = 1.0
    robust = flatten(t, f, window_length=w, method="biweight", return_trend=True)[1]
    mean = flatten(t, f, window_length=w, method="mean", return_trend=True)[1]
    sel = interior(t, w) & (np.abs(t[:, None] - t[flare][None, :]).min(axis=1) < 0.4)
    assert np.max(np.abs(robust[sel] - true[sel])) < 1e-3
    assert np.max(np.abs(mean[sel] - true[sel])) > 5e-4  # sanity: the mean is hurt


def test_beats_the_mean_on_contaminated_data():
    t, f, true = make_lc(n=3000, span=40.0, noise=5e-4)
    rng = np.random.default_rng(1)
    bad = rng.random(len(t)) < 0.08
    f = f.copy()
    f[bad] += rng.choice([-1, 1], bad.sum()) * 0.02
    sel = interior(t, 1.0)
    err = {}
    for m in ("biweight", "mean"):
        tr = flatten(t, f, window_length=1.0, method=m, return_trend=True)[1]
        err[m] = np.sqrt(np.mean((tr[sel] - true[sel]) ** 2))
    assert err["biweight"] < 0.5 * err["mean"]


def test_huge_cval_turns_the_biweight_into_the_mean():
    t, f, _ = make_lc(n=1000, span=20.0)
    bw = flatten(t, f, window_length=1.0, method="biweight", cval=1e7, return_trend=True)[1]
    mean = flatten(t, f, window_length=1.0, method="mean", return_trend=True)[1]
    ok = away_from_end(t, 1.0)
    np.testing.assert_allclose(bw[ok], mean[ok], atol=1e-9)


def test_small_cval_is_more_robust_than_large_cval():
    t, f, true = make_lc(n=3000, span=40.0)
    f = f.copy()
    f[::40] += 0.03
    sel = interior(t, 1.0)
    e = {}
    for c in (3.0, 20.0):
        tr = flatten(t, f, window_length=1.0, method="biweight", cval=c, return_trend=True)[1]
        e[c] = np.sqrt(np.mean((tr[sel] - true[sel]) ** 2))
    assert e[3.0] < e[20.0]


def test_discretised_flux_with_zero_mad_stays_finite():
    t = np.linspace(0, 10, 500)
    f = np.round(1 + 0.002 * np.sin(t), 2)  # many windows are all-identical (MAD == 0)
    trend = flatten(t, f, window_length=0.5, method="biweight", return_trend=True)[1]
    assert np.isfinite(trend).all()
    assert np.nanmax(np.abs(trend - f)) < 0.02


def test_time_units_do_not_matter():
    # same physical window in days and in hours -> identical trend
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 10, 600))
    f = 1 + 0.01 * np.sin(t) + rng.normal(0, 1e-3, 600)
    d = flatten(t, f, window_length=1.0, return_trend=True)[1]
    h = flatten(t * 24, f, window_length=24.0, return_trend=True)[1]
    np.testing.assert_allclose(d, h, atol=1e-12)


def test_result_is_independent_of_segments_far_away():
    t, f, _ = make_lc(n=1500, span=30.0)
    base = flatten(t, f, window_length=1.0, return_trend=True)[1]
    f2 = f.copy()
    f2[1000:] *= 3.0  # affects only t > 20
    other = flatten(t, f2, window_length=1.0, return_trend=True)[1]
    far = t < 20 - 1.0
    np.testing.assert_array_equal(base[far], other[far])


# ----------------------------------------------------- transit preservation
# Hippke et al. (2019), Sect. 4.6 + Conclusions: the optimal sliding window is
# ~3 x T14 ("as short as possible to remove variability, as long as necessary
# to preserve the transits"); for white noise w/T14 >~ 2.2 still preserves
# >~98% of the flux integral (Fig. 8). Much shorter windows eat the transit.
@pytest.fixture(scope="module")
def scen():
    return transit_scenario()


def test_window_3_t14_preserves_the_transit(scen):
    flat = flatten(scen["t"], scen["flux"], window_length=3 * scen["t14"], method="biweight")
    d = measured_depth(flat, scen["intransit"])
    assert d == pytest.approx(scen["depth"], rel=0.05)


def test_window_2p5_t14_still_preserves_the_transit_for_white_noise(scen):
    flat = flatten(scen["t"], scen["flux"], window_length=2.5 * scen["t14"], method="biweight")
    assert measured_depth(flat, scen["intransit"]) > 0.9 * scen["depth"]


def test_window_much_shorter_than_t14_removes_the_transit(scen):
    flat = flatten(scen["t"], scen["flux"], window_length=0.5 * scen["t14"], method="biweight")
    assert measured_depth(flat, scen["intransit"]) < 0.25 * scen["depth"]


def test_depth_retention_grows_monotonically_with_window(scen):
    depths = []
    for mult in (0.5, 1.0, 2.0, 3.0, 5.0):
        flat = flatten(scen["t"], scen["flux"], window_length=mult * scen["t14"],
                       method="biweight")
        depths.append(measured_depth(flat, scen["intransit"]))
    assert all(b >= a - 2e-4 for a, b in zip(depths, depths[1:]))


def test_transit_mask_rescues_a_short_window(scen):
    t, f, intr = scen["t"], scen["flux"], scen["intransit"]
    w = 1.0 * scen["t14"]
    unmasked = flatten(t, f, window_length=w, method="biweight")
    masked = flatten(t, f, window_length=w, method="biweight", mask=intr)
    assert measured_depth(unmasked, intr) < 0.3 * scen["depth"]
    assert measured_depth(masked, intr) == pytest.approx(scen["depth"], rel=0.05)


def test_flattened_transit_floor_is_at_one_minus_depth(scen):
    flat = flatten(scen["t"], scen["flux"], window_length=3 * scen["t14"], method="biweight")
    assert np.nanmedian(flat[scen["intransit"]]) == pytest.approx(1 - scen["depth"], abs=2e-4)
    assert np.nanmedian(flat[~scen["intransit"]]) == pytest.approx(1.0, abs=1e-4)
