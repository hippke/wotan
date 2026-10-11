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

Idea M (biweight iteration on running power sums of the sorted window, O(log n)
per step): implemented and dropped. Correct (2e-16 vs the plain iteration), but
slower: geomean 0.0284 s vs 0.0201 s. After the Aitken step only ~3.5 steps
remain, and the plain step is a vectorised (AVX-512) pass over a cache-resident
window; one power-sum pass costs about as much, plus scalar bookkeeping per step.

# All methods (round 2)

`python benchmarks/bench.py --method <m>`, window 0.5 d, same 2-core VM. Baseline:
master (64f77d7) with the same harness (`ref/<m>/baseline.json`). Datasets: Kepler-10
full and longest segment, plus white noise of the size in the table (pspline: no
kepler_full, ~220 s per run). Correctness: max |trend - master| over all datasets.

| Method | White N | master geomean | now | Speed-up | Max diff | Kept changes |
|---|---|---|---|---|---|---|
| biweight | 1M | 0.738 s | 0.0148 s | ~30-50x | 7e-10 | round 1 |
| median | 1M | 0.0922 s | 0.0041 s | 22x | 0 | sorted window (round 1), no MAD, no input copies, linear window bounds |
| mean | 1M | 0.0300 s | 0.0016 s | 19x | 1e-14 | running sums, linear window bounds, leaner result assembly |
| cosine | 100k | 0.379 s | 0.332 s | 1.14x | 1.1e-7 | design matrix by recurrence |
| lowess | 100k | 0.662 s | 0.078 s | 8.5x | 2e-11 | fused passes, prange, kernel once, 3 passes |
| rspline | 1M | 0.277 s | 0.136 s | 2.0x | 0 | vectorised cleaned_array, np.min/max, BSpline evaluation |
| hspline | 100k | 1.046 s | 0.076 s | 13.7x | 2.8e-7 | design_matrix, sparse Huber fit |
| cofiam | 100k | 0.141 s | 0.104 s | 1.36x | 0 | (flatten-level only) |
| pspline | 10k | 11.05 s | 8.59 s | 1.29x | 0 | cached pygam model matrix |

Current numbers (README table) are from a later run; cosine and cofiam now use the stable least squares (new references).

Timings of BLAS-heavy methods (cosine, cofiam, pspline) vary by ~10-20 % run to run.

Tried and dropped: median shift by binary search + loop (slower); cosine without
weighted copy (0 %), cosine lstsq in place via scipy (4.5 %, < 5 %); cofiam basis
built once and sliced (5 %, and |diff| 9e-7: numba's vectorised sin depends on the
loop length, and cofiam's Durbin-Watson choice reacts to ulp changes).
