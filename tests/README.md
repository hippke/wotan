# wotan test suite

    pip install -r tests/requirements-test.txt
    pytest                       # everything (~3 min on 2 cores; first run compiles numba code)
    pytest -m "not slow"         # skip statsmodels / pygam / GP / supersmoother based tests

Tests for methods whose optional dependency is missing (`statsmodels`, `scikit-learn`,
`pygam`, `supersmoother`, `astropy`) are skipped, not failed. The suite is offline and
deterministic (all random data use fixed seeds).

## Layout

| File | Content |
|---|---|
| `test_biweight.py`, `test_median.py` | priority sliders: trend recovery with *derived* tolerances, outlier rejection, mask, transit-depth preservation (paper Sect. 4.6: window ≈ 3×T14) |
| `test_cosine.py` | exact least-squares properties of `cosine` (+ `cofiam` smoke) |
| `test_huber.py` | `huber` vs `statsmodels` per window, median fallback, `hampel`, `ramsay` |
| `test_splines.py` | `rspline`, `hspline`, `pspline` |
| `test_lowess.py` | LOWESS identities and cross-check against `statsmodels` |
| `test_location_estimates.py` | the estimators themselves: equivariances, bounds, references |
| `test_slider.py` | window semantics vs brute force, edge cutoff, masks |
| `test_flatten_api.py` | return contract, NaN/gap handling, validation, documented defaults |
| `test_all_methods_smoke.py` | every method runs and tracks a known trend (low-priority methods) |
| `test_helpers_t14.py`, `test_slide_clip.py` | `transit_mask`, `cleaned_array`, `t14`, `slide_clip` |
| `test_bug_regressions.py` | one regression test per `BUGLOG.md` entry (all bugs are fixed) |
| `test_slider_equivalence.py` | the optimised sliders (sorted sliding window, running sums, parallel chunks) against a plain per-window numpy reference; `flatten` does not modify its inputs |
| `test_benchmark_references.py` | trends of the Kepler-10 benchmark light curve against the stored references in `benchmarks/ref` (1e-6, or 1e-3 of the noise) |
| `helpers.py` | synthetic light curves and **independent** reference implementations |

## How "correct" is decided

A detrender is not tested against its own output. Each test uses one of:

1. **A mathematical property of the definition** — e.g. a local-linear LOWESS reproduces
   straight lines; least-squares residuals are orthogonal to the basis; a location estimator
   is translation/scale/reflection equivariant and lies inside the data range.
2. **An independent reference** — brute-force windows, a re-implemented M-estimator,
   `scipy`, `statsmodels`, `astropy`.
3. **A physical expectation from the paper** — a window of ≈3×T14 preserves an injected
   transit, a window ≪T14 removes it.

Tolerances are derived, not tuned: e.g. a sliding window of length *w* attenuates a sinusoid
by `sinc(w/P)`, so the allowed trend error is `A(1 - sinc(w/P))` plus the averaging noise
(`helpers.sliding_mean_sine_bias`). Quantities such as "trend goes to infinity" are not
asserted anywhere: `flatten` divides by the trend, so behaviour near a zero trend is a
property of the definition (see `BUGLOG.md` BUG-025).

## Fixed bugs

`test_bug_regressions.py` asserts the *correct* behaviour for each of the 31 bugs in
`../BUGLOG.md` (GitHub issues #70-#100). The tests were written first as strict `xfail`
tests that failed on wotan 1.10, and pass after the fixes.

## TESS data (optional)

`wotan/tests.py` (the old regression script) needs two TESS light curves from MAST:

    sh tests/data/download_tess_data.sh

or, as a single command per file:

    curl -fLO https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits
    curl -fLO https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00207081058-s01_tess_v1_lc.fits

## numba cache

numba's on-disk cache does not track changes across files: after editing a compiled
function that another cached function calls, delete the cache before testing
(`find wotan -name "*.nb?" -delete`), otherwise the caller runs the old code.
