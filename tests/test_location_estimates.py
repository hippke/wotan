"""Unit tests of the robust location estimators in ``wotan.location_estimates``.

What is "correct" for a location estimator (all must hold unless noted):

* constant data -> that constant
* translation equivariance   T(x + c)  = T(x) + c
* scale equivariance         T(a x)    = a T(x)          (a > 0)
* reflection antisymmetry    T(-x)     = -T(x)
* permutation invariance
* result lies inside [min(x), max(x)]
* symmetric data -> centre of symmetry
* Gaussian data -> unbiased; robust ones resist gross outliers

Former bugs are covered in ``test_bug_regressions.py``.
"""
import numpy as np
import pytest
from scipy import stats

from wotan.location_estimates import (
    hampelfilt,
    hodges,
    huber_psi,
    location_iter,
    tau,
    trim_mean,
    winsorize_mean,
)

from helpers import m_estimator_reference

# name -> callable(float64 array) with wotan's default tuning constants
ESTIMATORS = {
    "biweight": lambda x: location_iter(x.copy(), 5.0, 1),
    "andrewsinewave": lambda x: location_iter(x.copy(), 1.339, 2),
    "welsch": lambda x: location_iter(x.copy(), 2.11, 3),
    "hodges": lambda x: hodges(x.copy()),
    "median": np.median,
    "mean": np.mean,
    "trim_mean": lambda x: trim_mean(x.copy(), 0.1),
    "winsorize": lambda x: winsorize_mean(x.copy(), 0.1),
    "hampelfilt": lambda x: hampelfilt(x.copy(), 3.0),
    "huber_psi": lambda x: huber_psi(x.copy(), 1.28),
    "tau": lambda x: tau(x.copy(), 4.5),
}
ROBUST = [k for k in ESTIMATORS if k != "mean"]
# estimators whose result is a (data-dependent) weighted average / order statistic
ALL = list(ESTIMATORS)

TOL = 3e-6  # FTOL is 1e-6 (absolute) for the iterative estimators


def gauss(n=301, seed=0, loc=0.0, scale=1.0):
    return np.random.default_rng(seed).normal(loc, scale, n)


# --------------------------------------------------------------- invariances
@pytest.mark.parametrize("name", [n for n in ALL if n not in ("tau", "huber_psi")])
def test_constant_data(name):
    x = np.full(25, 3.25)
    assert ESTIMATORS[name](x) == pytest.approx(3.25, abs=1e-12)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("shift", [1000.0, -37.5])
def test_translation_equivariance(name, shift):
    x = gauss(101, seed=1)
    assert ESTIMATORS[name](x + shift) == pytest.approx(ESTIMATORS[name](x) + shift, abs=TOL)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 2.0, 25.0])
def test_scale_equivariance(name, scale):
    # (the convergence tolerance is relative to the MAD: BUG-009)
    x = gauss(101, seed=2)
    assert ESTIMATORS[name](x * scale) == pytest.approx(
        ESTIMATORS[name](x) * scale, abs=TOL * scale
    )


@pytest.mark.parametrize("name", ALL)
def test_reflection_antisymmetry(name):
    x = gauss(101, seed=3) + 0.3 * np.abs(gauss(101, seed=4))  # skewed
    assert ESTIMATORS[name](-x) == pytest.approx(-ESTIMATORS[name](x), abs=TOL)


@pytest.mark.parametrize("name", ALL)
def test_permutation_invariance(name):
    rng = np.random.default_rng(5)
    x = gauss(80, seed=6)
    assert ESTIMATORS[name](x[rng.permutation(80)]) == pytest.approx(ESTIMATORS[name](x), abs=TOL)


@pytest.mark.parametrize("name", ALL)
def test_result_inside_data_range(name):
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(5, 60))
        x = rng.standard_t(2, n)  # heavy tails
        v = ESTIMATORS[name](x)
        assert x.min() - 1e-9 <= v <= x.max() + 1e-9


@pytest.mark.parametrize("name", ALL)
def test_symmetric_data_gives_center(name):
    half = np.abs(gauss(60, seed=8)) + 0.01
    x = np.concatenate([half, -half, [0.0]])
    assert ESTIMATORS[name](x) == pytest.approx(0.0, abs=TOL)


# ------------------------------------------------------- statistical behaviour
@pytest.mark.parametrize("name", ALL)
def test_unbiased_on_gaussian(name):
    x = gauss(2001, seed=9, loc=5.0, scale=2.0)
    # standard error of the mean = 2/sqrt(2001) = 0.045; allow ~7 SE
    assert ESTIMATORS[name](x) == pytest.approx(5.0, abs=0.35)


@pytest.mark.parametrize("name", ROBUST)
def test_resists_gross_outliers(name):
    x = np.concatenate([gauss(190, seed=10), np.full(10, 50.0)])  # 5% contamination
    assert abs(ESTIMATORS[name](x)) < 0.5
    assert abs(np.mean(x)) > 2.0  # sanity: the non-robust mean is ruined


# ------------------------------------------------------------ cross-checks
@pytest.mark.parametrize("n", [11, 50, 301])
@pytest.mark.parametrize(
    "name,code,cval,kind",
    [
        ("biweight", 1, 5.0, "biweight"),
        ("biweight", 1, 6.0, "biweight"),
        ("biweight", 1, 4.685, "biweight"),
        ("welsch", 3, 2.11, "welsch"),
        ("andrewsinewave", 2, 1.339, "andrew"),
    ],
)
def test_iterative_estimators_match_independent_reference(name, code, cval, kind, n):
    x = np.concatenate([gauss(n, seed=11), [8.0, 9.0]]) if n > 20 else gauss(n, seed=11)
    ours = location_iter(x.copy(), cval, code)
    ref = m_estimator_reference(x, cval, kind)
    assert ours == pytest.approx(ref, abs=TOL)


def test_biweight_matches_astropy_one_step_closely():
    astropy_stats = pytest.importorskip("astropy.stats")
    x = gauss(301, seed=12)
    # astropy takes a single biweight step from the median; wotan iterates.
    # For clean data both estimate the same quantity to well within the noise.
    assert location_iter(x.copy(), 5.0, 1) == pytest.approx(
        astropy_stats.biweight_location(x, c=5.0), abs=0.02
    )


@pytest.mark.parametrize("code", [1, 2, 3])
def test_large_cval_tends_to_the_mean(code):
    x = np.concatenate([gauss(100, seed=13), [6.0]])
    assert location_iter(x.copy(), 1e6, code) == pytest.approx(np.mean(x), abs=1e-6)


def test_biweight_mad_zero_returns_median():
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 9.0])  # MAD = 0
    assert location_iter(x.copy(), 5.0, 1) == 1.0


def test_hodges_lehmann_matches_brute_force_walsh_averages():
    x = gauss(23, seed=14)
    walsh = [(x[i] + x[j]) / 2 for i in range(len(x)) for j in range(i, len(x))]
    assert hodges(x.copy()) == pytest.approx(np.median(walsh), abs=1e-12)


@pytest.mark.parametrize("n", [3, 10, 25, 99, 100])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.25, 0.4])
def test_trim_mean_matches_scipy(n, p):
    x = gauss(n, seed=15)
    assert trim_mean(x.copy(), p) == pytest.approx(stats.trim_mean(x, p), abs=1e-12)


def test_hampelfilt_matches_definition():
    x = np.concatenate([gauss(60, seed=16), [20.0, -15.0]])
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    ref = np.mean(np.where(np.abs(x - med) >= 3.0 * mad, med, x))
    assert hampelfilt(x.copy(), 3.0) == pytest.approx(ref, abs=1e-12)


def test_hampelfilt_all_replaced_when_mad_zero_returns_median():
    x = np.array([2.0, 2.0, 2.0, 2.0, 7.0])
    assert hampelfilt(x.copy(), 3.0) == 2.0


def test_huber_psi_is_one_step_from_median_with_clipping_of_residuals():
    # Large cval -> nothing is clipped -> med + mad * mean((x-med)/mad) = mean
    x = gauss(50, seed=17)
    assert huber_psi(x.copy(), 1e9) == pytest.approx(np.mean(x), abs=1e-9)


def test_tau_is_finite_and_robust_for_continuous_data():
    x = np.concatenate([gauss(200, seed=18), [40.0, 45.0, -50.0]])
    v = tau(x.copy(), 4.5)
    assert np.isfinite(v)
    assert abs(v) < 0.5


def test_estimators_do_not_modify_their_input():
    # call the raw functions (no defensive .copy()) the way the sliders do
    x = gauss(40, seed=19)
    x0 = x.copy()
    raw = {
        "biweight": lambda d: location_iter(d, 5.0, 1),
        "andrewsinewave": lambda d: location_iter(d, 1.339, 2),
        "welsch": lambda d: location_iter(d, 2.11, 3),
        "hodges": hodges,
        "trim_mean": lambda d: trim_mean(d, 0.1),
        "winsorize": lambda d: winsorize_mean(d, 0.1),
        "hampelfilt": lambda d: hampelfilt(d, 3.0),
        "huber_psi": lambda d: huber_psi(d, 1.28),
        "tau": lambda d: tau(d, 4.5),
    }
    for name, fn in raw.items():
        fn(x)
        np.testing.assert_array_equal(x, x0, err_msg=name)
