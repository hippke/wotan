# Biweight performance (branch `performance`)

`python benchmarks/bench_biweight.py`, window 0.5 d, 2-core cloud VM, all commits
re-measured in one session with the same harness. Correctness: max |trend - master| <= 1e-6.

| Commit | Change | kepler_full (52k) | kepler_seg (1.9k) | white_1m (1M) | Geomean | Max diff |
|---|---|---|---|---|---|---|
| 64f77d7 | master (baseline) | 0.383 s | 13.3 ms | 52.8 s | 0.646 s | 0 |
| d651cec | fused iteration loop | | | | 0.213 s | 2e-16 |
| 18522b2 | parallel points (prange) | | | | 0.129 s | 2e-16 |
| 5e534d2 | sorted sliding window (median/MAD) | | | | 0.055 s | 2e-16 |
| 5528781 | vectorised trend copy-back | | | | 0.036 s | 2e-16 |
| 5bb7f48 | join segments once | | | | ~0.036 s (noise) | 2e-16 |
| 242dc52 | fix: running_segment cacheable again | | | | ~0.036 s (noise) | 2e-16 |
| 3c8cedb | Steffensen/Aitken acceleration | | | | 0.031 s | 7e-10 |
| eb4f437 | MAD by binary search | | | | 0.028 s | 7e-10 |
| b395e2b | swap leaving/entering in one move | 0.018 s | 1.0 ms | 0.64 s | 0.022 s | 7e-10 |

Total: ~29x geomean (22x Kepler full, 13x one segment, 83x 1M points) on 2 cores;
the parallel part scales with the core count.

Tried and dropped (< 5% or no gain): numba error_model="numpy" (3%), all segments
in one numba call (0%), branch-free biweight weights (0%), warm start from the
previous window (0%). Mask fast path: obsolete (no per-point copy any more).
