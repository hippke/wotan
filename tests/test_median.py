"""Sliding median: the most robust (and least efficient) time-windowed estimator."""
import numpy as np
import pytest

from wotan import flatten

from helpers import away_from_end, interior, make_lc, measured_depth, transit_scenario


def test_equals_numpy_median_of_the_window_on_an_even_grid():
    t = np.arange(0, 10, 0.01)
    f = 1 + np.random.default_rng(0).normal(0, 1e-2, len(t))
    window = 0.505  # 50.5 dt -> window = 51 samples exactly centred (no border ties)
    trend = flatten(t, f, window_length=window, method="median", break_tolerance=0,
                    return_trend=True)[1]
    half = 25
    sel = np.arange(half, len(t) - half - 1)
    ref = np.array([np.median(f[i - half : i + half + 1]) for i in sel])
    np.testing.assert_allclose(trend[sel], ref, atol=1e-12)


def test_preserves_a_step_edge_outside_half_a_window():
    t = np.linspace(0, 20, 2001)
    f = np.where(t < 10, 1.0, 2.0)
    w = 2.0
    trend = flatten(t, f, window_length=w, method="median", break_tolerance=0,
                    return_trend=True)[1]
    assert np.all(trend[t < 10 - w / 2 - 0.01] == 1.0)
    assert np.all(trend[(t > 10 + w / 2 + 0.01) & away_from_end(t, w)] == 2.0)
    mid = trend[np.abs(t - 10) < 0.01]
    assert 1.0 <= mid.min() and mid.max() <= 2.0


def test_ignores_isolated_spikes_exactly_on_a_flat_background():
    t = np.linspace(0, 10, 1000)
    f = np.ones_like(t)
    f[::50] += 1000.0
    f[25::50] -= 1000.0
    trend = flatten(t, f, window_length=0.5, method="median", return_trend=True)[1]
    ok = away_from_end(t, 0.5)
    np.testing.assert_array_equal(trend[ok], 1.0)


def test_close_to_biweight_for_gaussian_noise():
    t, f, _ = make_lc(n=3000, span=40.0)
    a = flatten(t, f, window_length=1.0, method="median", return_trend=True)[1]
    b = flatten(t, f, window_length=1.0, method="biweight", return_trend=True)[1]
    sel = interior(t, 1.0)
    # medians are ~25% noisier than the biweight; difference << amplitude of trend
    assert np.sqrt(np.mean((a[sel] - b[sel]) ** 2)) < 2e-4


def test_median_of_even_count_window_is_mean_of_middle_pair():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    f = np.array([10.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    # window 3.5 -> |dt| < 1.75: point t=2 sees t=1,2,3 (3 values) ; t=2.5 not sampled
    trend = flatten(t, f, window_length=3.5, method="median", break_tolerance=0,
                    return_trend=True)[1]
    assert trend[2] == 2.0  # median(1,2,3)
    assert trend[3] == 3.0  # median(2,3,4)


def test_survives_a_transit_at_three_t14():
    s = transit_scenario()
    flat = flatten(s["t"], s["flux"], window_length=3 * s["t14"], method="median")
    assert measured_depth(flat, s["intransit"]) == pytest.approx(s["depth"], rel=0.05)
