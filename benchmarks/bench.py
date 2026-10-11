"""Benchmark and correctness check for ``flatten(method=...)``.

Datasets (window_length 0.5 d for all methods)
  kepler_full : Kepler-10 (KIC 11904151) long cadence, all 15 quarter files
  kepler_seg  : the longest gap-free segment of kepler_full
  white_<N>   : white noise 1 + N(0, 1e-3), 2-min cadence, no gaps; N per method
                (benchmarks/data/white_1m_flux.npy, first N points)
Some methods are too slow for all datasets; they use a subset (see CONFIG).

Usage
  python benchmarks/bench.py --method median --save-ref   # reference (baseline)
  python benchmarks/bench.py --method median              # time + compare
Speed: best run after a JIT warm-up (>= 3 runs, or >= 20 s); score: geomean.
Correctness: max |trend - reference| <= 1e-6 and the same NaN pattern.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time as _time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_DIR = os.path.join(HERE, "ref")
WINDOW = 0.5
TOL = 1e-6

# method: (datasets, white size); chosen so that one baseline run takes <~ 15 s
CONFIG = {
    "biweight": (("kepler_full", "kepler_seg", "white"), 1_000_000),
    "median": (("kepler_full", "kepler_seg", "white"), 1_000_000),
    "mean": (("kepler_full", "kepler_seg", "white"), 1_000_000),
    "cosine": (("kepler_full", "kepler_seg", "white"), 100_000),
    "lowess": (("kepler_full", "kepler_seg", "white"), 100_000),
    "rspline": (("kepler_full", "kepler_seg", "white"), 1_000_000),
    "hspline": (("kepler_full", "kepler_seg", "white"), 100_000),
    "pspline": (("kepler_seg", "white"), 10_000),  # kepler_full: ~220 s per run
    "cofiam": (("kepler_full", "kepler_seg", "white"), 100_000),
}


def kepler_full():
    from astropy.io import fits
    files = sorted(glob.glob(os.path.join(HERE, "data", "kepler", "*_llc.fits")))
    ts, fs = [], []
    for fn in files:
        with fits.open(fn) as h:
            d = h[1].data
            t = np.asarray(d["TIME"], float)
            f = np.asarray(d["PDCSAP_FLUX"], float)
        ok = np.isfinite(t) & np.isfinite(f)
        ts.append(t[ok])
        fs.append(f[ok] / np.median(f[ok]))
    return np.concatenate(ts), np.concatenate(fs)


def datasets(names, white_n):
    from wotan.gaps import get_gaps_indexes
    t, f = kepler_full()
    out = {}
    if "kepler_full" in names:
        out["kepler_full"] = (t, f)
    if "kepler_seg" in names:
        g = get_gaps_indexes(t, break_tolerance=WINDOW / 2)
        k = int(np.argmax(np.diff(g)))
        a, b = g[k], min(g[k + 1], len(t))
        out["kepler_seg"] = (t[a:b], f[a:b])
    if "white" in names:
        fw = np.load(os.path.join(HERE, "data", "white_1m_flux.npy"))[:white_n]
        out["white_%dk" % (white_n // 1000)] = (np.arange(white_n) * (2.0 / 60 / 24), fw)
    return out


def run(method, t, f):
    from wotan import flatten
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return flatten(t, f, method=method, window_length=WINDOW, return_trend=True)[1]


def clear_numba_cache():
    """numba's ``cache=True`` does not track dependencies across files: a cached
    function keeps using an old version of a function it calls. Start from scratch."""
    for dirpath, _, files in os.walk(os.path.join(ROOT, "wotan")):
        for fn in files:
            if fn.endswith((".nbi", ".nbc")):
                os.remove(os.path.join(dirpath, fn))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="biweight")
    p.add_argument("--save-ref", action="store_true")
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    method = args.method
    names, white_n = CONFIG[method]
    ref_dir = os.path.join(REF_DIR, method)
    os.makedirs(ref_dir, exist_ok=True)
    clear_numba_cache()
    data = datasets(names, white_n)
    t0, f0 = next(iter(data.values()))
    run(method, t0[:3000], f0[:3000])  # JIT warm-up
    times, report = [], {}
    for name, (t, f) in data.items():
        best, k, total = np.inf, 0, 0.0
        while (k < args.repeat and total < 20) or (total < 1.0 and k < 200):
            s = _time.perf_counter()
            trend = run(method, t, f)
            dt = _time.perf_counter() - s
            best, k, total = min(best, dt), k + 1, total + dt
        ref_file = os.path.join(ref_dir, name + ".npy")
        if args.save_ref:
            np.save(ref_file, trend)
            diff = 0.0
        else:
            ref = np.load(ref_file)
            if not np.array_equal(np.isfinite(ref), np.isfinite(trend)):
                diff = np.inf
            else:
                both = np.isfinite(ref)
                diff = float(np.max(np.abs(ref[both] - trend[both]))) if both.any() else 0.0
        times.append(best)
        report[name] = dict(n=len(t), seconds=best, runs=k, max_abs_diff=diff,
                            ok=bool(diff <= TOL))
        print("%-8s %-12s n=%8d  %9.4f s  max|diff|=%.2e  %s"
              % (method, name, len(t), best, diff, "OK" if diff <= TOL else "FAIL"))
    code = ("import time,numpy as np;s=time.perf_counter();from wotan import flatten;"
            "t=np.linspace(0,10,2000);flatten(t,1+0.001*np.sin(t),method=%r,"
            "window_length=0.5);print(time.perf_counter()-s)" % method)
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                         check=True, capture_output=True, text=True).stdout
    startup = float(out.split()[-1])
    gm = float(np.exp(np.mean(np.log(times))))
    print("%-8s start-up (import + first call, cached): %.2f s" % (method, startup))
    print("%-8s geomean: %.4f s" % (method, gm))
    report["startup"] = startup
    report["geomean"] = gm
    out = os.path.join(ref_dir, "baseline.json" if args.save_ref else "last.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
