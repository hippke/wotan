# Benchmarks

    python benchmarks/bench.py --method biweight             # time + correctness check
    python benchmarks/bench.py --method biweight --save-ref  # store new references

`bench.py` times `flatten(method=..., window_length=0.5)` on up to three datasets and
compares the trends with stored references:

* `kepler_full`: Kepler-10 (KIC 11904151), all 15 long-cadence quarter files
  (`data/kepler`, PDCSAP flux; re-download with `data/download_kepler10.sh`)
* `kepler_seg`: its longest gap-free segment
* `white_<N>`: white noise `1 + N(0, 1e-3)` at 2-min cadence (`data/white_1m_flux.npy`),
  N per method (see `CONFIG` in `bench.py`; slow methods use fewer points)

Speed is the best of several runs after a JIT warm-up; the score is the geometric mean
over the datasets. The start-up time (import + first call, warm numba cache) is also
reported. Correctness: max |trend - reference| <= max(1e-6, 1e-3 * point-to-point noise)
and the same NaN pattern.

References (`ref/<method>/`): wotan 1.x (master 64f77d7), except `cosine` and `cofiam`,
whose least squares was made numerically stable in 2.0 (`ref/<method>/baseline.json`
keeps the 1.x timings). The script deletes numba's cache before each run (the cache does
not track changes across files).

Results: [RESULTS.md](RESULTS.md).
