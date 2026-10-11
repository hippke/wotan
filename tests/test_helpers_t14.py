"""``transit_mask``, ``cleaned_array`` and ``t14``."""
import numpy as np
import pytest
from scipy import constants as sc

from wotan import t14, transit_mask
from wotan.helpers import cleaned_array


# ------------------------------------------------------------------ transit_mask
def test_transit_mask_counts_on_a_uniform_grid():
    t = np.arange(0, 100, 0.01)
    m = transit_mask(t, period=10.0, duration=0.5, T0=5.0)
    assert m.dtype == bool and m.shape == t.shape
    # 10 transits, each 0.5 d long = 50 samples (open interval, +-1 sample)
    assert abs(m.sum() - 10 * 50) <= 10


def test_transit_mask_is_periodic_and_centred_on_T0():
    P, D, T0 = 3.7, 0.4, 1.25
    centres = T0 + P * np.arange(-3, 8)
    assert transit_mask(centres, P, D, T0).all()
    assert not transit_mask(centres + P / 2, P, D, T0).any()


def test_transit_mask_edges_are_exclusive():
    P, D, T0 = 5.0, 1.0, 0.0
    inside = np.array([0.499999, -0.499999, 5.2])
    outside = np.array([0.500001, -0.500001, 2.5])
    assert transit_mask(inside, P, D, T0).all()
    assert not transit_mask(outside, P, D, T0).any()


def test_transit_mask_T0_may_lie_outside_the_data_and_times_may_be_negative():
    t = np.linspace(-20, -10, 1001)
    m = transit_mask(t, period=4.0, duration=0.2, T0=1000.0)
    assert m.any()
    expect = np.abs(((t - 1000.0 + 2.0) % 4.0) - 2.0) < 0.1
    np.testing.assert_array_equal(m, expect)


def test_transit_mask_nan_times_are_never_in_transit_and_do_not_warn(recwarn):
    t = np.array([np.nan, 5.0, np.nan, 5.05, 7.0])
    m = transit_mask(t, 10.0, 0.5, 5.0)
    np.testing.assert_array_equal(m, [False, True, False, True, False])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_transit_mask_longer_than_period_is_everything():
    t = np.linspace(0, 10, 50)
    assert transit_mask(t, period=1.0, duration=1.01, T0=0.3).all()


# ------------------------------------------------------------------ cleaned_array
def test_cleaned_array_drops_nan_inf_and_none_keeping_order():
    t = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=object)
    y = np.array([10.0, None, 30.0, 40.0, np.inf, 60.0], dtype=object)
    ct, cy = cleaned_array(t, y)
    np.testing.assert_array_equal(ct, [1.0, 4.0, 6.0])
    np.testing.assert_array_equal(cy, [10.0, 40.0, 60.0])
    assert ct.dtype == cy.dtype == float


def test_cleaned_array_with_uncertainties():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, np.nan, 3.0, 4.0])
    dy = np.array([0.1, 0.1, np.nan, 0.4])
    ct, cy, cdy = cleaned_array(t, y, dy)
    np.testing.assert_array_equal(ct, [1.0, 4.0])
    np.testing.assert_array_equal(cy, [1.0, 4.0])
    np.testing.assert_array_equal(cdy, [0.1, 0.4])


def test_cleaned_array_of_clean_input_is_identity_and_empty_stays_empty():
    t, y = np.arange(5.0), np.arange(5.0) ** 2
    ct, cy = cleaned_array(t, y)
    np.testing.assert_array_equal(ct, t)
    np.testing.assert_array_equal(cy, y)
    et, ey = cleaned_array(np.array([]), np.array([]))
    assert et.size == ey.size == 0


# ---------------------------------------------------------------------- t14
def test_t14_regression_values_from_the_original_script():
    # values asserted in wotan/tests.py (Sun-Earth analogue, P = 365 d)
    assert t14(R_s=1, M_s=1, P=365) == pytest.approx(0.6490025258902046, abs=1e-8)
    assert t14(R_s=1, M_s=1, P=365, small_planet=True) == pytest.approx(0.5403690143737738, abs=1e-8)


def test_t14_earth_transit_of_the_sun_lasts_about_13_hours():
    hours = 24 * t14(R_s=1, M_s=1, P=365.25, small_planet=True)
    assert 12.5 < hours < 13.5


def test_t14_jupiter_analogue_lasts_about_30_hours():
    hours = 24 * t14(R_s=1, M_s=1, P=4332.6, small_planet=True)
    assert 28.0 < hours < 31.0


@pytest.mark.parametrize("R_s,M_s,P", [(1.0, 1.0, 10.0), (1.6, 1.7, 100.0), (0.3, 0.3, 2.0)])
def test_t14_matches_the_central_circular_orbit_formula(R_s, M_s, P):
    G = sc.G
    M = M_s * 1.989e30
    a = (G * M * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    R = R_s * 6.95508e8
    exact = P / np.pi * np.arcsin(R / a)  # days; the code uses the small-angle form
    assert t14(R_s, M_s, P, small_planet=True) == pytest.approx(exact, rel=2e-3)


def test_t14_scalings():
    base = t14(1.0, 1.0, 10.0, small_planet=True)
    assert t14(2.0, 1.0, 10.0, small_planet=True) == pytest.approx(2 * base)  # ~ R_s
    assert t14(1.0, 1.0, 80.0, small_planet=True) == pytest.approx(2 * base)  # ~ P^(1/3)
    assert t14(1.0, 8.0, 10.0, small_planet=True) == pytest.approx(base / 2)  # ~ M^(-1/3)


def test_t14_large_planet_assumption_is_longer():
    assert t14(1, 1, 10) > t14(1, 1, 10, small_planet=True)
