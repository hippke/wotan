import warnings

import numpy as np
import pytest

from helpers import make_lc


@pytest.fixture(scope="session")
def lc():
    """Default synthetic light curve: sinusoidal trend, white noise, 30 d."""
    return make_lc()


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _quiet_numpy_warnings():
    # The library divides by zero / takes NaN means in documented edge cases.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def pytest_collection_modifyitems(config, items):
    # Tag everything that touches a slow optional backend so `-m "not slow"` works.
    slow_names = ("pspline", "hampel", "ramsay", "gp_", "huber_slider", "supersmoother")
    for item in items:
        if any(s in item.name for s in slow_names) and "slow" not in item.keywords:
            item.add_marker(pytest.mark.slow)
