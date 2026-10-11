"""The trends on the Kepler-10 benchmark light curve must agree with the stored
references (benchmarks/ref: wotan 1.x, master 64f77d7; cosine and cofiam: 2.0) to
max(1e-6, 1e-3 * noise)."""
import os
import sys
import warnings

import numpy as np
import pytest

from wotan import flatten

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "benchmarks")
sys.path.insert(0, BENCH)
pytest.importorskip("astropy")
bench = pytest.importorskip("bench")

CASES = [(m, d) for m in ("biweight", "median", "mean", "cosine", "lowess", "rspline",
                          "hspline", "cofiam") for d in ("kepler_full", "kepler_seg")]
CASES.append(("pspline", "kepler_seg"))


@pytest.mark.parametrize("method,dataset", CASES)
def test_trend_matches_the_1x_reference(method, dataset):
    if method == "pspline":
        pytest.importorskip("pygam")
    ref_file = os.path.join(BENCH, "ref", method, dataset + ".npy")
    if not os.path.exists(ref_file) or not os.path.isdir(os.path.join(BENCH, "data", "kepler")):
        pytest.skip("benchmark data not available")
    t, f = bench.datasets((dataset,), 0)[dataset]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trend = flatten(t, f, method=method, window_length=bench.WINDOW, return_trend=True)[1]
    ref = np.load(ref_file)
    np.testing.assert_array_equal(np.isfinite(trend), np.isfinite(ref))
    ok = np.isfinite(ref)
    assert np.max(np.abs(trend[ok] - ref[ok])) <= bench.tolerance(f)
