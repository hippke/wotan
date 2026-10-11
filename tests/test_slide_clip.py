"""``slide_clip``: sliding time-windowed outlier clipper (documented behaviour that
does not depend on the method/center keyword mix-up of BUG-006 or the end effect
of BUG-007; those are in test_known_bugs.py)."""
import numpy as np
import pytest

from wotan import slide_clip


def lc(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20, n)
    return t, 1 + rng.normal(0, 1e-3, n)


def test_output_has_input_length_and_input_is_untouched():
    t, f = lc()
    f0 = f.copy()
    out = slide_clip(t, f, window_length=1.0)
    assert out.shape == f.shape
    np.testing.assert_array_equal(f, f0)


def test_clipped_points_are_nan_and_the_rest_keep_their_values():
    t, f = lc()
    f = f.copy()
    f[300] += 0.05
    f[600] -= 0.05
    out = slide_clip(t, f, window_length=1.0, low=3, high=3)
    assert np.isnan(out[300]) and np.isnan(out[600])
    kept = ~np.isnan(out)
    np.testing.assert_array_equal(out[kept], f[kept])
    assert kept.sum() > 0.95 * len(f)


def test_high_and_low_thresholds_act_on_their_own_side():
    t, f = lc()
    f = f.copy()
    f[300] += 0.05
    f[600] -= 0.05
    only_high = slide_clip(t, f, window_length=1.0, low=1e6, high=3)
    only_low = slide_clip(t, f, window_length=1.0, low=3, high=1e6)
    assert np.isnan(only_high[300]) and not np.isnan(only_high[600])
    assert np.isnan(only_low[600]) and not np.isnan(only_low[300])


def test_looser_thresholds_clip_fewer_points():
    t, f = lc()
    tight = np.isnan(slide_clip(t, f, 1.0, low=2, high=2)).sum()
    loose = np.isnan(slide_clip(t, f, 1.0, low=5, high=5)).sum()
    assert loose <= tight


def test_clipping_is_local_not_global():
    # A step in level must not cause the points of the upper plateau to be clipped
    t = np.linspace(0, 20, 1000)
    f = np.where(t < 10, 1.0, 1.5) + np.random.default_rng(1).normal(0, 1e-3, 1000)
    out = slide_clip(t, f, window_length=1.0, low=5, high=5)
    away = (np.abs(t - 10) > 1.0) & (t < 19.5)
    assert not np.isnan(out[away]).any()
