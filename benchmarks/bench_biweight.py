"""Benchmark and correctness check for ``flatten(method="biweight")``.

Datasets
  1. kepler_full : all long-cadence quarters of a Kepler LC (FITS files in
                   benchmarks/data/kepler/*_llc.fits (Kepler-10, KIC 11904151, Q0-Q14)); Kepler-like synthetic stand-in if absent
  2. kepler_seg  : the longest gap-free segment of dataset 1
  3. white_1m    : synthetic white noise, 1e6 points, 2-min cadence, no gaps

Usage
  python benchmarks/bench_biweight.py --save-ref   # store reference trends (baseline)
  python benchmarks/bench_biweight.py              # time + compare against reference
Speed is the best of ``--repeat`` runs (after a JIT warm-up); the score is the geomean.
"""
import argparse
import glob
import json
import os
import time as _time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_DIR = os.path.join(HERE, "ref")  # reference trends of the baseline (master)
WINDOW = 0.5
TOL = 1e-6


def kepler_full():
    files = sorted(glob.glob(os.path.join(HERE, "data", "kepler", "*_llc.fits")))
    if files:
        from astropy.io import fits
        ts, fs = [], []
        for fn in files:
            with fits.open(fn) as h:
                d = h[1].data
                t = np.asarray(d["TIME"], float)
                f = np.asarray(d["PDCSAP_FLUX"], float)
            ok = np.isfinite(t) & np.isfinite(f)
            ts.append(t[ok])
            fs.append(f[ok] / np.median(f[ok]))
        return np.concatenate(ts), np.concatenate(fs), "real (%d files)" % len(files)
    # Kepler-like stand-in: 17 quarters of 29.4-min cadence, monthly downlink gaps
    rng = np.random.default_rng(42)
    dt = 29.4 / 60 / 24
    ts = []
    t0 = 0.0
    for q in range(17):
        for m in range(3):
            n = int(30 / dt)
            ts.append(t0 + np.arange(n) * dt)
            t0 += n * dt + 1.0
        t0 += 2.0
    t = np.concatenate(ts)
    f = 1 + 1e-3 * np.sin(2 * np.pi * t / 12.3) + rng.normal(0, 1e-4, t.size)
    return t, f, "synthetic stand-in"


def datasets():
    from wotan.gaps import get_gaps_indexes
    t, f, kind = kepler_full()
    g = get_gaps_indexes(t, break_tolerance=WINDOW / 2)
    lens = np.diff(g)
    k = int(np.argmax(lens))
    a, b = g[k], min(g[k + 1], len(t))
    n = 1_000_000
    tw = np.arange(n) * (2.0 / 60 / 24)
    white_file = os.path.join(HERE, "data", "white_1m_flux.npy")
    if os.path.exists(white_file):
        fw = np.load(white_file)
    else:
        fw = 1 + np.random.default_rng(1).normal(0, 1e-3, n)
        np.save(white_file, fw)
    return {
        "kepler_full": (t, f, kind),
        "kepler_seg": (t[a:b], f[a:b], kind),
        "white_1m": (tw, fw, "synthetic"),
    }


def run(t, f):
    from wotan import flatten
    return flatten(t, f, method="biweight", window_length=WINDOW, return_trend=True)[1]


def clear_numba_cache():
    """numba's ``cache=True`` does not track dependencies across files: a cached
    ``running_segment`` keeps using the old ``location_iter``. Start from scratch."""
    for dirpath, _, files in os.walk(os.path.join(ROOT, "wotan")):
        for fn in files:
            if fn.endswith((".nbi", ".nbc")):
                os.remove(os.path.join(dirpath, fn))


def main():
    clear_numba_cache()
    p = argparse.ArgumentParser()
    p.add_argument("--save-ref", action="store_true")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--only", default=None)
    args = p.parse_args()
    os.makedirs(REF_DIR, exist_ok=True)
    data = datasets()
    run(*[x[:2000] for x in data["kepler_seg"][:2]])  # JIT warm-up
    times, report = [], {}
    for name, (t, f, kind) in data.items():
        if args.only and name != args.only:
            continue
        best, k, total = np.inf, 0, 0.0
        # at least ``repeat`` runs; fast datasets are repeated for >= 1 s (less noise)
        while k < args.repeat or (total < 1.0 and k < 200):
            s = _time.perf_counter()
            trend = run(t, f)
            dt = _time.perf_counter() - s
            best, k, total = min(best, dt), k + 1, total + dt
        ref_file = os.path.join(REF_DIR, name + ".npy")
        if args.save_ref:
            np.save(ref_file, trend)
            diff = 0.0
        else:
            ref = np.load(ref_file)
            both = np.isfinite(ref) & np.isfinite(trend)
            if not np.array_equal(np.isfinite(ref), np.isfinite(trend)):
                diff = np.inf
            else:
                diff = float(np.max(np.abs(ref[both] - trend[both])))
        times.append(best)
        report[name] = dict(n=len(t), kind=kind, seconds=best, max_abs_diff=diff,
                            ok=bool(diff <= TOL))
        print("%-12s n=%8d  %9.4f s  max|diff|=%.2e  %s  [%s]"
              % (name, len(t), best, diff, "OK" if diff <= TOL else "FAIL", kind))
    # Start-up: import + first call in a fresh process, with numba's cache already warm
    # (catches functions that silently stop being cacheable)
    import subprocess, sys
    code = ("import time,numpy as np;s=time.perf_counter();from wotan import flatten;"
            "t=np.linspace(0,10,2000);flatten(t,1+0*t,window_length=0.5);"
            "print(time.perf_counter()-s)")
    env = dict(os.environ, PYTHONPATH=ROOT)
    startup = float(subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                   capture_output=True, text=True).stdout.split()[-1])
    print("start-up (import + first call, cached): %.2f s" % startup)
    report["startup"] = startup
    gm = float(np.exp(np.mean(np.log(times))))
    print("geomean: %.4f s" % gm)
    report["geomean"] = gm
    out = os.path.join(REF_DIR, "baseline.json" if args.save_ref else "last.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
