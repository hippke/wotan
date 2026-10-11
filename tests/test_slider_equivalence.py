"""The optimised sliders (sorted sliding window, parallel chunks) must match a plain
per-window reference (median/MAD from numpy, the biweight-type iteration in numpy)."""
import numpy as np
import pytest

from wotan import flatten


def reference(t, f, mask, w, cval, method):
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        sel = (t >= t[i] - w / 2) & (t < t[i] + w / 2) & mask
        d = f[sel]
        if len(d) == 0:
            continue
        med = np.median(d)
        if method == "median":
            out[i] = med
            continue
        mad = np.median(np.abs(d - med))
        if mad == 0:
            out[i] = med
            continue
        c, delta = med, np.inf
        for _ in range(1000):
            if abs(delta) <= 1e-6 * mad:
                break
            dist = d - c
            u = dist / (cval * mad)
            if method == "biweight":
                wt = (1 - u**2) ** 2
                wt[np.abs(u) >= 1] = 0
            elif method == "welsch":
                wt = np.exp(-u**2 / 2)
                wt[np.abs(u) >= 1] = 0
            else:  # andrewsinewave
                u[u == 0] = 1e-10
                wt = np.sin(u) / u
                wt[np.abs(u) >= np.pi] = 0
            new = c + np.sum(dist * wt) / np.sum(wt)
            delta, c = c - new, new
        out[i] = c
    return out


@pytest.mark.parametrize("method,cval", [("biweight", 5), ("welsch", 2.11),
                                         ("andrewsinewave", 1.339), ("median", None)])
@pytest.mark.parametrize("masked", [False, True])
def test_sliding_window_matches_reference(method, cval, masked):
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 10, 1500))
    f = 1 + 0.01 * np.sin(t) + rng.normal(0, 1e-3, t.size)
    f[::37] = np.round(f[::37], 3)  # some exact ties
    f[500:510] -= 0.05  # outliers
    mask = np.zeros(t.size, bool)
    if masked:
        mask[(t > 3) & (t < 3.4)] = True  # TLS convention: True = masked
    kw = {} if cval is None else {"cval": cval}
    trend = flatten(t, f, method=method, window_length=0.7, return_trend=True,
                    mask=mask if masked else None, break_tolerance=5, **kw)[1]
    ref = reference(t, f, ~mask, 0.7, cval, method)
    # the iteration stops within 1e-6 * MAD of the fixed point: allow a few of those
    np.testing.assert_allclose(trend, ref, atol=1e-8, rtol=0)


def test_sliding_window_with_edge_cutoff_and_gaps():
    rng = np.random.default_rng(8)
    t = np.concatenate([np.linspace(0, 3, 400), np.linspace(4, 9, 600)])
    f = 1 + rng.normal(0, 1e-3, t.size)
    a = flatten(t, f, method="biweight", window_length=0.5, edge_cutoff=0.2,
                return_trend=True)[1]
    ref = np.full(t.size, np.nan)
    for seg in (slice(0, 400), slice(400, 1000)):
        ts, fs = t[seg], f[seg]
        r = reference(ts, fs, np.ones(ts.size, bool), 0.5, 5, "biweight")
        keep = (ts >= ts.min() + 0.2) & (ts <= ts.max() - 0.2)
        r[~keep] = np.nan
        ref[seg] = r
    np.testing.assert_allclose(a, ref, atol=1e-8, rtol=0)


def test_median_mad_of_sorted_array_matches_numpy():
    from wotan.slider import _median_mad_sorted
    rng = np.random.default_rng(0)
    for n in list(range(1, 40)) * 50 + [360, 361]:
        x = np.sort(np.round(rng.normal(0, 1, n), rng.integers(0, 4)))  # with ties
        med, mad = _median_mad_sorted(x)
        assert med == pytest.approx(np.median(x), abs=1e-12)
        assert mad == pytest.approx(np.median(np.abs(x - np.median(x))), abs=1e-12)


@pytest.mark.parametrize("method", ["biweight", "median", "mean", "hampelfilt", "tau",
                                    "cosine", "cofiam", "lowess", "rspline", "hspline",
                                    "savgol", "medfilt"])
def test_flatten_does_not_modify_its_inputs(method):
    """flatten works on views of the input when nothing has to be removed"""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 5, 500)
    f = 1 + rng.normal(0, 1e-3, t.size)
    f[100:105] -= 0.02
    mask = np.zeros(t.size, bool)
    mask[200:220] = True
    t0, f0, m0 = t.copy(), f.copy(), mask.copy()
    wl = 51 if method in ("savgol", "medfilt") else 0.5
    kw = {} if method in ("savgol", "medfilt") else {"mask": mask}
    flatten(t, f, method=method, window_length=wl, robust=True, **kw)
    np.testing.assert_array_equal(t, t0)
    np.testing.assert_array_equal(f, f0)
    np.testing.assert_array_equal(mask, m0)
