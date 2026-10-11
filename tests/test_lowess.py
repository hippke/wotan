"""LOWESS (Cleveland 1979): local linear regression with tricube distance weights
and iterated bisquare robustness weights (c = 6 x median |residual|; Hippke 2019,
Sect. 2.5).  Window = ``window_length`` in time units -> fraction of the points.

Definition-based facts:

* local *linear* regression reproduces straight lines (and constants) exactly
* affine equivariance  L(a y + b) = a L(y) + b
* agrees with statsmodels' lowess for the same span
* robust to outliers; masked points get zero weight
"""
import numpy as np
import pytest

from wotan import flatten

from helpers import interior, make_lc, measured_depth, transit_scenario


@pytest.mark.parametrize("window", [0.5, 2.0, 7.0])
def test_reproduces_a_straight_line_exactly(window):
    t = np.linspace(0, 10, 400)
    f = 1 + 0.02 * t
    tr = flatten(t, f, window_length=window, method="lowess", return_trend=True)[1]
    np.testing.assert_allclose(tr, f, atol=1e-9)


def test_reproduces_a_constant():
    t = np.linspace(0, 10, 400)
    tr = flatten(t, np.full(400, 5.0), window_length=2.0, method="lowess",
                 return_trend=True)[1]
    np.testing.assert_allclose(tr, 5.0, atol=1e-9)


@pytest.mark.parametrize("window", [1.5, 3.0])
def test_matches_statsmodels_lowess_in_the_interior(window):
    sm = pytest.importorskip("statsmodels.nonparametric.smoothers_lowess")
    t, f, _ = make_lc(n=600, span=12.0, seed=3)
    tr = flatten(t, f, window_length=window, method="lowess", return_trend=True)[1]
    ref = sm.lowess(f, t, frac=window / 12.0, it=20, return_sorted=False)
    sel = slice(100, -100)
    np.testing.assert_allclose(tr[sel], ref[sel], atol=1e-4)  # measured 4.6e-5 / 3e-8


def test_follows_the_trend(lc):
    t, f, true = lc
    w = 1.0
    tr = flatten(t, f, window_length=w, method="lowess", return_trend=True)[1]
    assert np.max(np.abs(tr - true)[interior(t, w)]) < 6e-4  # measured 2.8e-4


def test_is_affine_equivariant(lc):
    t, f, _ = lc
    a = flatten(t, f, window_length=1.5, method="lowess", return_trend=True)[1]
    b = flatten(t, 2.0 * f + 5.0, window_length=1.5, method="lowess", return_trend=True)[1]
    np.testing.assert_allclose(b, 2.0 * a + 5.0, atol=1e-9)


def test_is_robust_to_flares():
    t, f, true = make_lc(n=1500, span=30.0)
    f = f.copy()
    idx = np.arange(50, 1500, 100)
    f[idx] += 0.05
    tr = flatten(t, f, window_length=2.0, method="lowess", return_trend=True)[1]
    mean = flatten(t, f, window_length=2.0, method="mean", return_trend=True)[1]
    sel = interior(t, 2.0)
    assert np.sqrt(np.mean((tr - true)[sel] ** 2)) < 0.5 * np.sqrt(np.mean((mean - true)[sel] ** 2))


def test_mask_removes_the_bias_of_a_wide_dip():
    t = np.linspace(0, 10, 1000)
    f = 1 + np.random.default_rng(2).normal(0, 2e-4, 1000)
    dip = (t > 5.0) & (t < 5.8)
    f[dip] -= 0.01
    centre = (t > 5.3) & (t < 5.5)
    plain = flatten(t, f, window_length=1.0, method="lowess", return_trend=True)[1]
    masked = flatten(t, f, window_length=1.0, method="lowess", mask=dip,
                     return_trend=True)[1]
    assert abs(np.mean(plain[centre] - 1)) > 5e-3
    assert abs(np.mean(masked[centre] - 1)) < 0.1 * abs(np.mean(plain[centre] - 1))


def test_output_is_finite_and_complete(lc):
    t, f, _ = lc
    tr = flatten(t, f, window_length=1.0, method="lowess", return_trend=True)[1]
    assert tr.shape == t.shape and np.isfinite(tr).all()


def test_time_offset_sensitivity_stays_small():
    t, f, _ = make_lc(n=600, span=20.0)
    a = flatten(t, f, window_length=4.0, method="lowess", return_trend=True)[1]
    b = flatten(t + 2457000.0, f, window_length=4.0, method="lowess", return_trend=True)[1]
    assert np.max(np.abs(a - b)) < 1e-4  # measured 1.7e-5


def test_window_scaling_in_time_units_not_samples():
    # Same physical window on twice the sampling -> same trend (up to noise averaging)
    rng = np.random.default_rng(0)
    t1 = np.linspace(0, 20, 800)
    t2 = np.linspace(0, 20, 1600)
    true1 = 1 + 0.01 * np.sin(2 * np.pi * t1 / 8)
    true2 = 1 + 0.01 * np.sin(2 * np.pi * t2 / 8)
    a = flatten(t1, true1, window_length=3.0, method="lowess", return_trend=True)[1]
    b = flatten(t2, true2, window_length=3.0, method="lowess", return_trend=True)[1]
    np.testing.assert_allclose(np.interp(t1, t2, b)[100:-100], a[100:-100], atol=1e-4)  # measured 3e-5: int() rounding of the window in samples


@pytest.fixture(scope="module")
def scen():
    return transit_scenario(cadence_min=10.0)


def test_window_3_t14_preserves_the_transit(scen):
    flat = flatten(scen["t"], scen["flux"], window_length=3 * scen["t14"], method="lowess")
    assert measured_depth(flat, scen["intransit"]) == pytest.approx(scen["depth"], rel=0.05)


def test_window_half_t14_absorbs_the_transit(scen):
    flat = flatten(scen["t"], scen["flux"], window_length=0.5 * scen["t14"], method="lowess")
    assert measured_depth(flat, scen["intransit"]) < 0.2 * scen["depth"]


def test_noise_free_data_does_not_collapse_the_robustness_weights():
    """Exactly constant or linear data: the residuals are round-off; they must not
    be used as the robust scale (all iterations, all points finite)."""
    from wotan.lowess import lowess
    t = np.linspace(0, 10, 400)
    for y in (np.full(400, 5.0), 1 + 0.02 * t):
        for maxiter in (1, 10, 30):
            tr = lowess(t, y, np.ones(400), 2.0, 6.0, maxiter)
            np.testing.assert_allclose(tr, y, atol=1e-9)
