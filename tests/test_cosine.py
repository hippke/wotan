"""``cosine``: least-squares fit of ``1, t, sin(2 pi t k/D), cos(2 pi t k/D)``, k=1..degree.

Definition (``wotan.cofiam``): ``degree = int(span / window_length)`` and the
basis period scale is ``D = 2 * span``. The fit is *linear least squares*, so
exact algebraic properties can be tested: exact recovery of any signal in the
span of the basis, orthogonality of the residuals to every basis function,
reduction to a straight line for degree 0, and weighted LS for masks.
``robust=True`` additionally sigma-clips dips and refits.
"""
import numpy as np
import pytest

from wotan import flatten
from wotan.cofiam import detrend_cosine, matrix_gen

from helpers import make_lc


def basis_signal(t, coefs):
    """A signal exactly inside the cosine basis (degree = len(coefs) - 1 harmonics)."""
    dur = 2 * (t.max() - t.min())
    f = 1.0 + 0.002 * t / (t.max() - t.min())
    for k, (a, b) in enumerate(coefs, start=1):
        f = f + a * np.sin(2 * np.pi * t * k / dur) + b * np.cos(2 * np.pi * t * k / dur)
    return f


def test_exact_recovery_of_a_signal_in_the_basis_span():
    t = np.linspace(0, 12, 400)
    f = basis_signal(t, [(0.01, 0.004), (-0.003, 0.002), (0.001, -0.0015)])
    trend = flatten(t, f, window_length=4.0, method="cosine", return_trend=True)[1]
    # degree = int(12/4) = 3 -> exactly the harmonics present
    np.testing.assert_allclose(trend, f, atol=1e-9)


def test_degree_is_span_over_window_and_sets_the_basis_size():
    t = np.linspace(0, 10, 200)
    assert matrix_gen(t, 0).shape == (200, 2)
    assert matrix_gen(t, 3).shape == (200, 8)
    m = matrix_gen(t, 2)
    np.testing.assert_array_equal(m[:, 0], 1.0)
    np.testing.assert_array_equal(m[:, 1], t)
    dur = 2 * (t.max() - t.min())
    np.testing.assert_allclose(m[:, 2], np.sin(2 * np.pi * t * 1 / dur))
    np.testing.assert_allclose(m[:, 3], np.cos(2 * np.pi * t * 1 / dur))
    np.testing.assert_allclose(m[:, 4], np.sin(2 * np.pi * t * 2 / dur))
    np.testing.assert_allclose(m[:, 5], np.cos(2 * np.pi * t * 2 / dur))


def test_residuals_are_orthogonal_to_the_basis_normal_equations(lc):
    t, f, _ = lc
    window = 6.0
    trend = flatten(t, f, window_length=window, method="cosine", return_trend=True)[1]
    degree = int((t.max() - t.min()) / window)
    resid = f - trend
    cols = matrix_gen(t, degree)
    scale = np.linalg.norm(cols, axis=0) * np.linalg.norm(resid)
    assert np.max(np.abs(cols.T @ resid) / scale) < 1e-8


def test_window_longer_than_span_is_a_straight_line_fit(lc):
    t, f, _ = lc
    trend = flatten(t, f, window_length=1e3, method="cosine", return_trend=True)[1]
    ref = np.polyval(np.polyfit(t, f, 1), t)
    np.testing.assert_allclose(trend, ref, atol=1e-9)


def test_more_flexible_with_shorter_windows(lc):
    t, f, true = lc
    err = {}
    for w in (15.0, 6.0, 3.0):
        tr = flatten(t, f, window_length=w, method="cosine", return_trend=True)[1]
        err[w] = np.sqrt(np.mean((tr - true) ** 2))
    assert err[3.0] < err[15.0]
    assert err[6.0] < err[15.0]


def test_follows_the_sinusoid_to_noise_level(lc):
    t, f, true = lc
    trend = flatten(t, f, window_length=4.0, method="cosine", return_trend=True)[1]
    assert np.max(np.abs(trend - true)) < 4e-4  # noise 5e-4, ~(2*degree+2)=16 params / 1500


def test_mask_means_weighted_least_squares_on_the_unmasked_points():
    t = np.linspace(0, 20, 800)
    rng = np.random.default_rng(3)
    f = 1 + 0.01 * np.sin(2 * np.pi * t / 15) + rng.normal(0, 5e-4, 800)
    f[300:340] -= 0.02
    drop = np.zeros(800, bool)
    drop[300:340] = True
    window = 4.0
    trend = flatten(t, f, window_length=window, method="cosine", mask=drop,
                    return_trend=True)[1]
    degree = int((t.max() - t.min()) / window)
    X = matrix_gen(t, degree)
    beta = np.linalg.lstsq(X[~drop], f[~drop], rcond=None)[0]
    np.testing.assert_allclose(trend, X @ beta, atol=1e-9)


def test_masking_a_transit_removes_the_bias_it_causes():
    t = np.linspace(0, 20, 800)
    true = 1 + 0.01 * np.sin(2 * np.pi * t / 15)
    f = true + np.random.default_rng(4).normal(0, 5e-4, 800)
    dip = (t > 9) & (t < 11)
    f[dip] -= 0.02
    plain = flatten(t, f, window_length=4.0, method="cosine", return_trend=True)[1]
    masked = flatten(t, f, window_length=4.0, method="cosine", mask=dip, return_trend=True)[1]
    assert abs(np.mean(plain[dip] - true[dip])) > 10 * abs(np.mean(masked[dip] - true[dip]))
    assert abs(np.mean(masked[dip] - true[dip])) < 2e-4


def test_robust_clipping_removes_dip_bias_without_a_mask():
    t = np.linspace(0, 20, 800)
    true = 1 + 0.01 * np.sin(2 * np.pi * t / 15)
    f = true + np.random.default_rng(4).normal(0, 5e-4, 800)
    dip = (t > 9) & (t < 11)
    f[dip] -= 0.02
    plain = flatten(t, f, window_length=4.0, method="cosine", return_trend=True)[1]
    robust = flatten(t, f, window_length=4.0, method="cosine", robust=True,
                     return_trend=True)[1]
    assert abs(np.mean(robust[dip] - true[dip])) < 0.1 * abs(np.mean(plain[dip] - true[dip]))


def test_robust_equals_plain_fit_when_there_are_no_outliers_large_enough_to_clip():
    t, f, _ = make_lc(n=600, span=20.0, noise=0.0)  # noiseless: nothing to clip
    a = flatten(t, f, window_length=5.0, method="cosine", return_trend=True)[1]
    b = flatten(t, f, window_length=5.0, method="cosine", robust=True, return_trend=True)[1]
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_user_mask_is_not_modified_by_the_robust_iterations():
    t, f, _ = make_lc(n=500, span=20.0)
    f = f.copy()
    f[200:220] -= 0.02
    m = np.zeros(500, bool)
    m[10:20] = True
    m0 = m.copy()
    flatten(t, f, window_length=4.0, method="cosine", robust=True, mask=m)
    np.testing.assert_array_equal(m, m0)


def test_detrend_cosine_direct_call_matches_flatten():
    t, f, _ = make_lc(n=500, span=20.0)
    direct = detrend_cosine(t, f, 4.0, False, np.ones(500))
    via = flatten(t, f, window_length=4.0, method="cosine", return_trend=True)[1]
    np.testing.assert_allclose(direct, via, rtol=1e-12)


def test_time_offset_sensitivity_stays_small():
    # BJD-like offsets worsen the conditioning (columns 1 and t); see BUGLOG note N-2
    t, f, _ = make_lc(n=600, span=20.0)
    a = flatten(t, f, window_length=4.0, method="cosine", return_trend=True)[1]
    b = flatten(t + 2457000.0, f, window_length=4.0, method="cosine", return_trend=True)[1]
    assert np.max(np.abs(a - b)) < 1e-4


def test_cofiam_runs_and_follows_the_trend(lc):
    t, f, true = lc
    trend = flatten(t, f, window_length=3.0, method="cofiam", return_trend=True)[1]
    assert np.isfinite(trend).all()
    assert np.max(np.abs(trend - true)) < 1e-3
