"""Huber-type robust sliders: ``huber`` (iterative, statsmodels), ``hampel`` and
``ramsay`` (statsmodels RLM), plus the relationship of ``huber`` to ``huber_psi``.

Definition of ``huber`` in wotan: ``statsmodels.robust.scale.Huber(c=cval)`` jointly
estimates location and scale of the window, with ``cval=1.5`` by default; if it does
not converge, the window median is used (issues #6 and #19).
"""
import numpy as np
import pytest

from wotan import flatten

from helpers import away_from_end, interior, make_lc, window_reference

sm_scale = pytest.importorskip("statsmodels.robust.scale")
pytest.importorskip("statsmodels.api")


def huber_ref(c=1.5):
    return lambda x: float(sm_scale.Huber(maxiter=1000, tol=1e-6, c=c)(x)[0])


@pytest.fixture(scope="module")
def irregular():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 10, 300))
    f = 1 + rng.normal(0, 1e-2, 300) + 0.02 * np.sin(t)
    return t, f


def test_huber_slider_equals_statsmodels_huber_on_each_window(irregular):
    t, f = irregular
    trend = flatten(t, f, window_length=1.0, method="huber", break_tolerance=0,
                    return_trend=True)[1]
    ref = window_reference(t, f, 1.0, huber_ref())
    ok = away_from_end(t, 1.0)
    np.testing.assert_allclose(trend[ok], ref[ok], atol=1e-10)


@pytest.mark.parametrize("cval", [1.0, 1.5, 3.0])
def test_cval_is_the_huber_tuning_constant(irregular, cval):
    t, f = irregular
    trend = flatten(t, f, window_length=1.0, method="huber", cval=cval, break_tolerance=0,
                    return_trend=True)[1]
    ref = window_reference(t, f, 1.0, huber_ref(cval))
    ok = away_from_end(t, 1.0)
    np.testing.assert_allclose(trend[ok], ref[ok], atol=1e-10)


def test_default_cval_is_1p5(irregular):
    t, f = irregular
    a = flatten(t, f, window_length=1.0, method="huber")
    b = flatten(t, f, window_length=1.0, method="huber", cval=1.5)
    np.testing.assert_array_equal(a, b)


def test_huber_resists_outliers_better_than_the_mean():
    t, f, true = make_lc(n=1500, span=30.0)
    f = f.copy()
    f[::37] += 0.05
    sel = interior(t, 1.0)
    err = {}
    for m in ("huber", "mean"):
        tr = flatten(t, f, window_length=1.0, method=m, return_trend=True)[1]
        err[m] = np.sqrt(np.mean((tr[sel] - true[sel]) ** 2))
    assert err["huber"] < 0.5 * err["mean"]


def test_masked_points_are_ignored(irregular):
    t, f = irregular
    mask = (t > 4.2) & (t < 4.8)
    f = f.copy()
    f[mask] += 3.0
    trend = flatten(t, f, window_length=1.0, method="huber", mask=mask,
                    break_tolerance=0, return_trend=True)[1]
    ref = window_reference(t, f, 1.0, huber_ref(), mask_keep=~mask)
    ok = away_from_end(t, 1.0)
    np.testing.assert_allclose(trend[ok], ref[ok], atol=1e-10)


def test_falls_back_to_the_median_when_the_iteration_fails(irregular, monkeypatch):
    # Issues #6/#19: non-convergence happens; wotan promises the median instead.
    class Boom:
        def __init__(self, *a, **k):
            pass

        def __call__(self, x):
            raise RuntimeError("did not converge")

    monkeypatch.setattr(sm_scale, "Huber", Boom)
    t, f = irregular
    trend = flatten(t, f, window_length=1.0, method="huber", break_tolerance=0,
                    return_trend=True)[1]
    ref = window_reference(t, f, 1.0, np.median)
    ok = away_from_end(t, 1.0)
    np.testing.assert_allclose(trend[ok], ref[ok], atol=1e-12)


def test_window_with_a_single_point_gives_nan():
    t = np.array([0.0, 0.1, 0.2, 5.0, 9.8, 9.9, 10.0, 10.1])
    f = np.ones(8) + np.arange(8) * 1e-3
    trend = flatten(t, f, window_length=0.5, method="huber", break_tolerance=0,
                    return_trend=True)[1]
    assert np.isnan(trend[3])  # alone in its window


def test_huber_and_huber_psi_agree_closely_on_gaussian_data():
    t, f, _ = make_lc(n=1000, span=20.0)
    a = flatten(t, f, window_length=1.0, method="huber", return_trend=True)[1]
    b = flatten(t, f, window_length=1.0, method="huber_psi", return_trend=True)[1]
    sel = interior(t, 1.0)
    assert np.sqrt(np.mean((a[sel] - b[sel]) ** 2)) < 1.5e-4  # noise/sqrt(n_window)


@pytest.mark.parametrize("method", ["huber", "hampel", "ramsay"])
def test_linear_ramp_is_reproduced_in_the_interior(method):
    dt, m = 0.02, 8
    t = np.arange(0, 10, dt)
    f = 1.0 + 0.05 * t
    window = (2 * m + 0.5) * dt
    trend = flatten(t, f, window_length=window, method=method, break_tolerance=0,
                    return_trend=True)[1]
    sel = interior(t, window + 2 * dt)
    np.testing.assert_allclose(trend[sel], f[sel], atol=1e-6)


@pytest.mark.parametrize("method,cval", [("hampel", (1.7, 3.4, 8.5)), ("ramsay", 0.3)])
def test_hampel_and_ramsay_defaults_and_robustness(method, cval):
    t, f, true = make_lc(n=400, span=10.0)
    f = f.copy()
    f[::23] += 0.05
    a = flatten(t, f, window_length=1.0, method=method, return_trend=True)[1]
    b = flatten(t, f, window_length=1.0, method=method, cval=cval, return_trend=True)[1]
    np.testing.assert_array_equal(a, b)  # defaults are the documented ones
    mean = flatten(t, f, window_length=1.0, method="mean", return_trend=True)[1]
    sel = interior(t, 1.0)
    assert np.sqrt(np.mean((a[sel] - true[sel]) ** 2)) < np.sqrt(np.mean((mean[sel] - true[sel]) ** 2))


def test_hampel_accepts_the_25A_constants():
    t, f, true = make_lc(n=300, span=10.0)
    tr = flatten(t, f, window_length=1.0, method="hampel", cval=(2.5, 4.5, 9.5),
                 return_trend=True)[1]
    assert np.isfinite(tr).all()
    assert np.max(np.abs(tr - true)[interior(t, 1.0)]) < 1e-3
