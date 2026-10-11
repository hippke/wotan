# wotan BUGLOG

Bugs found while building the first test suite (`tests/`).  Every entry has a
test in `tests/test_known_bugs.py` (same number) that asserts the *correct*
behaviour and is marked `xfail(strict=True)`: it fails today, and turns the run
red the moment the bug is fixed so the marker (and this entry) get removed.

    pytest tests/test_known_bugs.py --runxfail      # see all of them fail

* **Code under test:** wotan 1.10, commit `7752703` (`master`, "update")
* **Environment:** Python 3.13.16, numpy 2.5.3, scipy 1.16.3, numba 0.68.0,
  scikit-learn 1.9.1, statsmodels 0.15.0, pygam 0.12.0, supersmoother 0.4, astropy 8.0.1
* **How "correct" was decided:** the method's own definition (Tukey, Cleveland, Huber,
  Kipping+2013, ...), exact algebraic properties (e.g. local-linear LOWESS reproduces
  lines, least-squares residuals are orthogonal to the basis), independent references
  (scipy, statsmodels, astropy, brute-force re-implementations), the wotan paper
  (Hippke et al. 2019, arXiv:1906.00966) and the repo docs.
* **Severity:** **High** = silently wrong output or a crash on common input;
  **Medium** = wrong/crashing on less common input or a documented feature that does not
  work; **Low** = cosmetic, API, docs, robustness.
* **Confidence** is given where I am less than certain about the *intended* behaviour.

## Summary

GitHub issues: BUG-001 … BUG-031 are issues [#70](https://github.com/hippke/wotan/issues/70) … [#100](https://github.com/hippke/wotan/issues/100) (issue number = 69 + bug number). Demonstration figures: `docs/bug_figures/`.

| ID | Sev | Where | One-line summary |
|----|-----|-------|------------------|
| [BUG-001](#bug-001) | High | `regression.py:39` | `ridge`/`lasso`/`elasticnet` crash on current scikit-learn (`max_iter=1e5` is a float) |
| [BUG-002](#bug-002) | High | `slider.py:54-59,128-132` | The last sample of every segment is never in any window |
| [BUG-003](#bug-003) | High | `slider.py:43-44,60` | Flux ≤ 0 is silently dropped; *masked* negative flux leaks into the window |
| [BUG-004](#bug-004) | Medium | `location_estimates.py:67-85` | `tau` raises `ZeroDivisionError` when the window MAD is 0 |
| [BUG-005](#bug-005) | Medium | `location_estimates.py:126-134` | `winsorize` is wrong: extra point, asymmetric tails, out-of-bounds read for n=1 |
| [BUG-006](#bug-006) | Medium | `slide_clipper.py:8,15` | `slide_clip` keywords never match: defaults silently use std/mean instead of MAD/median |
| [BUG-007](#bug-007) | Low | `slide_clipper.py:66,74` | `slide_clip` always returns NaN for the last sample |
| [BUG-008](#bug-008) | Medium | `gp.py:80,108` | `method='gp'` with the documented default `kernel=None` crashes; unknown kernel → `UnboundLocalError` |
| [BUG-009](#bug-009) | Medium | `location_estimates.py:33`, `lowess.py:48` | Absolute convergence tolerances: results depend on flux units |
| [BUG-010](#bug-010) | Medium | `cofiam.py:61-62` | `cofiam` never varies the number of harmonics: it is a plain full-degree cosine fit |
| [BUG-011](#bug-011) | Medium | `pspline.py:25,41-46` | `stdev_cut` doubles as the iteration count; convergence break only runs if `verbose` |
| [BUG-012](#bug-012) | Low | `flatten.py:331-336` | `return_nsplines=True` without `return_trend=True` returns `None` |
| [BUG-013](#bug-013) | Medium | `flatten.py` | Unsorted `time` is not detected: silently wrong trend and NaNs |
| [BUG-014](#bug-014) | Medium | `location_estimates.py:75-82` | `tau`: wrong outlier cut-off (outliers get weight up to 4.5) and non-standard scale |
| [BUG-015](#bug-015) | Low | `location_estimates.py:97` | `huber_psi`: Newton step divides by *n* instead of the inlier count |
| [BUG-016](#bug-016) | Low | `flatten.py:261` | `supersmoother` span uses the whole light curve, not the segment |
| [BUG-017](#bug-017) | Medium | `huber_spline.py:46` | `hspline` returns all-NaN, no warning, when `window_length > span/4` |
| [BUG-018](#bug-018) | Low | `flatten.py` | No input validation: `window_length≤0/NaN`, `(n,1)` flux, `mask` as list, empty input |
| [BUG-019](#bug-019) | Low | `cofiam.py:50,53`, `gp.py:71,74` | Library prints to stdout unconditionally (`cosine` always prints "Converged.") |
| [BUG-020](#bug-020) | Low | `iter_spline.py:30-33` | `rspline` convergence check is dead code: always 10 spline fits |
| [BUG-021](#bug-021) | Low | `flatten.py` docstring | 5 parameters undocumented (incl. `mask`, issue #55); typos; wrong default |
| [BUG-022](#bug-022) | Low | `lowess.py:10` | Tricube radius uses `x[-2]`: last window point gets non-zero weight at the left edge |
| [BUG-023](#bug-023) | Low | `norms.py` | Module runs demo code on import and uses `np.alltrue` (removed in NumPy 2.0) |
| [BUG-024](#bug-024) | Medium | `flatten.py` | `mask` is silently ignored by `pspline`, `cofiam`, `supersmoother`, `savgol`, `medfilt` (+ regressions) |
| [BUG-025](#bug-025) | Medium | `flatten.py` | `flatten` divides by the trend: zero-crossing / mean-subtracted flux gives wild ratios, no warning (issue #64) |
| [BUG-026](#bug-026) | Low | `cofiam.py:matrix_gen` | `cosine`/`cofiam` are not invariant to a time offset (BJD): ill-conditioned basis |
| [BUG-027](#bug-027) | Low | `huber_spline.py` | `hspline` precision depends on the flux offset (absolute optimiser tolerance) |
| [BUG-028](#bug-028) | Medium | `cofiam.py`, `iter_spline.py` | Robust clipping is one-sided: flares are not rejected, only dips |
| [BUG-029](#bug-029) | Low | `slider.py` | An isolated one-sample segment silently gets a NaN trend (confidence: low) |
| [BUG-030](#bug-030) | Low | 8 places | Bare `except:` swallows `KeyboardInterrupt`/`SystemExit` |
| [BUG-031](#bug-031) | Medium | `flatten.py` | `inf` in the flux is not treated as invalid and corrupts the trend |

## Scorecard for the priority detrenders

What the new tests verified to be **correct**, and which bugs touch each method.

| Method | Verified correct (test) | Bugs that touch it |
|---|---|---|
| **biweight** | Matches an independent iterate-to-convergence M-estimator to 3e-6 (= `FTOL`) for c = 4.685/5/6; within 0.02 of astropy's one-step biweight; window semantics equal a brute-force `abs(t-tᵢ)<w/2` reference; translation/scale (moderate)/reflection equivariance; large `cval` → mean; flare/outlier rejection; **paper Sect. 4.6: window 3×T14 keeps ≥95 % of an injected depth, 0.5×T14 destroys it**; transit mask rescues short windows | 002, 003, 009, 013, 018 |
| **median** | Equals `np.median` of the window on even and irregular grids; step edges preserved; isolated spikes rejected exactly; linear ramp reproduced | 002, 003, 013 |
| **cosine** | Exact least squares: recovers any signal in the basis span to 1e-9, residuals orthogonal to all basis columns, degree 0 = straight line, `mask` = weighted LS on the kept points, `robust` clipping removes dip bias, user mask not mutated | 019 (prints), 026, 028 |
| **huber** | Equals `statsmodels…Huber(c=cval)` on every window to 1e-10; masked points ignored; falls back to the window median on non-convergence (issues #6/#19) | 002 |
| **splines** | `rspline` reproduces cubics to 1e-10, clips dips, mask ≡ dropping the points; `hspline` tracks the trend, scale-equivariant at realistic flux, issue #64 not reproducible as an hspline bug; `pspline` tracks, `nsplines` grows with wiggliness, `edge_cutoff` exact, issue #57 fixed | 011, 012, 017, 020, 024, 027, 028 |
| **lowess** | Reproduces constants and straight lines to 1e-9, affine equivariant, agrees with `statsmodels` lowess to 4.6e-5 (interior), mask removes wide-dip bias, 3×T14 keeps the transit | 009, 022 |

---

## BUG-001
**ridge / lasso / elasticnet crash with current scikit-learn** — High · `regression.py:39`

```python
flatten(t, f, window_length=3.0, method="ridge")
# InvalidParameterError: The 'max_iter' parameter of Ridge must be an int in the range [1, inf) or None. Got 100000.0 instead.
```
`max_iter = 1e5` is a float. Observed with scikit-learn 1.9.1. `.travis.yml` pins
`scikit-learn==0.20`, which is why CI never saw it.  **Fix:** `max_iter = int(1e5)`.
With that one-line fix all three run (default `cval=1` fits weakly: max error ≈ signal amplitude — a tuning question, not logged as a bug).
**Test:** `test_bug001_*` (non-strict: depends on the installed scikit-learn).

## BUG-002
**The last sample of each segment is never part of any window** — High · `slider.py:54-59` (numba sliders), `slider.py:128-132` (`huber`/`hampel`/`ramsay`), `slide_clipper.py:74`

```python
flatten([0, 1, 2], [1, 1, 100], window_length=10, method="mean", break_tolerance=0, return_trend=True)[1]
# -> [1. 1. 1.]      expected [34. 34. 34.]
```
`idx_end` is advanced with `... and idx_end < size-1` and then used as an *exclusive*
slice bound, so index `size-1` can never be included. Consequences: (1) the last point's
own trend value ignores its own flux; (2) every window that reaches the end of the
segment is missing its last sample (on a 400-point irregular test, the 20 points within
w/2 of the end disagree with the brute-force definition); (3) a 2-point segment returns
`trend[1] == flux[0]`. Silent, and it affects every light curve and every segment after a gap.
**Fix:** `while idx_end < size and time[idx_end] < time[i] + half_window: idx_end += 1`.
**Tests:** `test_bug002_*` (3). The remaining slider tests deliberately exclude windows touching the end (`helpers.away_from_end`).

## BUG-003
**Flux ≤ 0 is silently discarded; masked negative flux leaks in** — High · `slider.py:43-44,60`

`running_segment` hides masked points by multiplying their flux by −1 and then keeps
only `f > -1e-12`.  Any *genuinely* non-positive flux is therefore dropped, and a masked
point with negative flux becomes positive and is **included**.
```python
flatten(t, np.full(100, -5.0), window_length=2.0, method="median")        # NaN trend (every window empty)
flatten(np.linspace(0,10,100), np.arange(100.)-50, window_length=1., method="median")  # 47/100 NaN
# masked flux -100 inside a -5 baseline: trend there is +100, expected -5
```
The statsmodels-based sliders use NaN masking and are *not* affected, so `median` and
`huber` disagree on the same data. Mitigation in practice: wotan assumes flux ≈ 1
(issue #56 comment), but nothing warns when that is violated. Related: #56, #64.
**Fix:** pass the mask as a separate boolean array and select `flux[idx_start:idx_end][mask[...] > 0]`
(NaN masking is unreliable under `fastmath=True`, which is presumably why the sign trick was used).
**Tests:** `test_bug003_*` (3, plus a positive-flux control that passes).

## BUG-004
**`tau` raises `ZeroDivisionError` when the window MAD is 0** — Medium · `location_estimates.py:67-85`

```python
tau(np.array([1., 1., 1., 1., 2., 100.]), 4.5)    # ZeroDivisionError
flatten(t, flux_with_flat_stretches, window_length=1.0, method="tau")   # crashes the whole call
```
`biweight`, `huber_psi` and `hampelfilt` all guard `MAD == 0`; `tau` does not. Realistic for quantised or saturated/flat flux.
**Fix:** `if mad == 0: return med`. **Tests:** `test_bug004_*` (2).

## BUG-005
**`winsorize` does not winsorize correctly** — Medium · `location_estimates.py:126-134`

Three defects in one function: (a) `idx = int(p*n) + 1` winsorizes one point more per
tail than `scipy.stats.mstats.winsorize` (`int(p*n)`); (b) `sorted[-idx:] = sorted[-idx]`
fills the upper tail with a value *from inside* the replaced block while the lower tail
uses the first value *outside* it → the two tails are treated differently and the
estimator is not reflection-antisymmetric (`W(-x) ≠ -W(x)`), which biases skewed data;
(c) for a 1-sample window `sorted_data[1]` reads out of bounds (returns `6.9e-310`
garbage), and for n = 2 it returns the maximum.

| n | wotan | scipy |
|---|---|---|
| 2 | 0.8216 | 0.5836 |
| 10 | −0.0997 | −0.1506 |
| 101 | −0.0345 | −0.0360 |

(10 % cut, random Gaussian samples.)  **Fix:** `k = int(p*n)`; `sorted[:k] = sorted[k]`; `sorted[n-k:] = sorted[n-k-1]`.
**Tests:** `test_bug005_*` (10). The symmetry tests in `test_location_estimates.py` exclude `winsorize` with a pointer to this entry.

## BUG-006
**`slide_clip` method/center keywords never match** — Medium · `slide_clipper.py:8,15`

The docstring offers `method='mad'|'std'` and `center='median'|'mean'`, but `clipit`
tests `center == 'mad'` and `method == 'median'`.  The documented defaults
(`mad`/`median`) therefore fall through to the `else` branches and clip with **std
around the mean**; `method='mad'` and `'std'` give identical output.
On a 400-point test the default clips 15 points, the true MAD/median rule 31 (the buggy
keywords `method='median', center='mad'` reproduce the correct 31).
Note: the pinned value `948.9926368754939` in `wotan/tests.py` encodes this behaviour and must be updated with the fix.
**Fix:** compare against `'mad'`/`'median'` (and validate the keywords). **Tests:** `test_bug006_*` (2).

## BUG-007
**`slide_clip` always returns NaN for the last sample** — Low · `slide_clipper.py:66,74`

Same exclusive-window cause as BUG-002 plus `range(size-1)`: index `size-1` is never written, so it stays at its `np.nan` initialisation even for perfectly clean data.
**Test:** `test_bug007_*`.

## BUG-008
**`method='gp'` with the default `kernel=None` crashes** — Medium · `gp.py:80,108`

`flatten` documents `kernel : default squared_exp` and `make_gp` has `if kernel is None or kernel == 'squared_exp'`, but then `'periodic' in kernel` raises
`TypeError: argument of type 'NoneType' is not iterable`. An unknown kernel string leaves `trend_segment` unbound (`UnboundLocalError`).
**Fix:** `kernel = kernel or 'squared_exp'`; raise `ValueError` for unknown kernels. **Tests:** `test_bug008_*` (2).

## BUG-009
**Absolute convergence tolerances make results depend on the flux unit** — Medium · `location_estimates.py:33` (`FTOL = 1e-6`), `lowess.py:48` (`ftol=1e-6`)

A location estimator must be scale-equivariant, `T(a·x) = a·T(x)`. The biweight iterates until the centre moves by < 1e-6 *absolute*; for flux ~1e-3 it stops after one step:

| flux scale | relative deviation of the biweight vs. scale 1 |
|---|---|
| 1 | 0 (converged value is within 1.6e-7) |
| 1e-3, 1e-6, 1e-9 | 8.8e-5 (identical at all three: it is the *one-step* estimate) |

88 ppm is larger than the ~10 ppm jitter the README says the iteration removes.
`lowess` shows the same at scale 1e-6 (2.6e-5 relative). Normalised flux (≈1) is unaffected, flux in physical units or small mags is not.
**Fix:** make the tolerance relative (`FTOL * mad` or `FTOL * abs(center)`).
**Tests:** `test_bug009_*` (2); the same properties at moderate scales pass in `test_location_estimates.py`.

## BUG-010
**`cofiam` never changes the number of harmonics** — Medium · `cofiam.py:61-62`

```python
for k_m in range(1, degree + 1):
    matrix = matrix_gen(t, degree)      # <- should be k_m
```
Kipping et al. (2013) fit increasing numbers of harmonics and keep the one that minimises the Durbin–Watson statistic. Here the basis is always built with `degree`, so the DW comparison never changes anything and `cofiam` is identical to the non-robust `cosine` (observed: `matrix_gen` called 11× with degree 10 only).
**Fix:** `matrix_gen(t, k_m)`. **Test:** `test_bug010_*`.

## BUG-011
**`pspline`: sigma threshold used as the iteration count** — Medium · `pspline.py:25,41-46`

`for i in range(stdev_cut)` (default 2): `stdev_cut=2.5` → `TypeError: 'float' object cannot be interpreted as an integer`, `stdev_cut=3` quietly means 3 iterations, `1` means 1. The convergence `break` sits inside `if verbose:`, so with `verbose=False` it never fires.
**Fix:** iterate `range(constants.PSPLINES_MAXITER)` and move the `break` out of `if verbose`. **Test:** `test_bug011_*`.

## BUG-012
**`return_nsplines=True` alone returns `None`** — Low · `flatten.py:331-336`

Branches cover (trend, nsplines), (trend, no nsplines), (no trend, no nsplines) but not (no trend, nsplines): falls off the end. **Test:** `test_bug012_*`.

## BUG-013
**Unsorted `time` is not detected** — Medium · `flatten.py`

The sliders and gap detection assume ascending time. A random permutation of a 300-point light curve produced 39 NaNs and trend errors up to 2.0e-2 (the signal amplitude is 1e-2), with no error or warning.
**Fix:** sort internally and un-sort the result, or raise `ValueError`. **Test:** `test_bug013_*`.

## BUG-014
**`tau`: wrong outlier cut-off and scale** — Medium · `location_estimates.py:75-82` · *confidence: high for the cut-off, medium for the scale*

```python
s = 0.675 * mad
temp = (1 - (y / cval)**2)**2
if abs(temp) <= cval:          # compares a weight with the tuning constant
```
The weight `(1-(y/c)²)²` is meant to vanish for `|y/c| > 1`; the test above instead admits any point whose weight is ≤ `cval` (4.5), i.e. all points up to `|y/c| ≈ 1.77`, **with weights up to 4.5 — outliers outweigh the bulk (max weight of a good point is 1)**. Separately, a normal-consistent scale is `MAD / 0.6745`, not `MAD × 0.675` (factor ≈ 2.2 smaller).
On N(0,1) plus 8 points at +3.3 the estimate moves by 0.52; with the cut-off fixed it moves by 0.03, with both fixed by 0.09.
The wotan paper (Sect. 2.4.2) does not define the estimator, so this rests on Yohai & Zamar (1988) / Maronna & Zamar (2002) and the inconsistency within the code.
**Fix:** `if abs(y) <= cval:` (or weight 0 beyond) and `s = mad / 0.6745`. **Test:** `test_bug014_*`.

## BUG-015
**`huber_psi` Newton step uses the wrong denominator** — Low · `location_estimates.py:97` · *confidence: medium*

`b = (abs(y) <= cval).sum()` is evaluated **after** clipping `y` to ±cval, so it always equals *n*. The comment ("how many values in y < cval?") and the Newton step `T₁ = T₀ + s·Σψ(y)/Σψ′(y)` call for the number of *unclipped* points. On 100 N(0,1) + 20 outliers at +10: wotan 0.3274, Newton step 0.3534.
The paper does not define this estimator (Sect. 2.4 only gives a pseudo-Huber loss), so confirm the intent. **Fix:** count before clipping. **Test:** `test_bug015_*`.

## BUG-016
**`supersmoother` span is computed from the whole light curve** — Low · `flatten.py:261`

`win = window_length / (max(time) - min(time))` uses the full `time` array (also fragile when `time` contains NaN: Python's `max`), not the current segment, so segments are not independent: a segment detrended alone differs from the same segment inside a longer light curve by 2.6e-3 (biweight: exactly 0).
**Fix:** use `time_view`. **Test:** `test_bug016_*`.

## BUG-017
**`hspline` silently returns an all-NaN trend** — Medium · `huber_spline.py:46`

`len(knots) < 4` → `np.full(len(time), nan)` with no warning. Happens whenever `window_length > segment_span/4` (e.g. window 4 d on a 10 d light curve: 300/300 NaN), a very natural mistake after segmentation by `break_tolerance`.
**Fix:** warn or raise. **Test:** `test_bug017_*`.

## BUG-018
**Missing input validation** — Low · `flatten.py`

* `window_length` ≤ 0, NaN, or 1e-9 → all-NaN output, no error.
* Flux of shape `(n, 1)` (e.g. `DataFrame[col].to_numpy()` in recent pandas/yfinance; plausibly the cause of issue #54) → `MaskError: data size is 300, mask size is 90000` from an (n, n) broadcast in `isnan(time*flux)`.
* `mask` as a Python list → `TypeError: bad operand type for unary ~: 'list'`.
* Empty or all-NaN input → `ValueError: zero-size array to reduction operation minimum which has no identity` (numpy's message, not wotan's).

**Tests:** `test_bug018_*` (5). Behaviour that is already fine (unknown method, bad `proportiontocut`, length mismatch) is covered in `test_flatten_api.py`.

## BUG-019
**The library prints to stdout** — Low · `cofiam.py:50,53`, `gp.py:71,74`

`detrend_cosine` prints `Converged.` on **every** call, including the default non-robust one (`converged = True` on entry). Robust cosine and robust GP print one line per iteration regardless of `verbose`. **Fix:** gate on `verbose` or use `logging`. **Test:** `test_bug019_*`.

## BUG-020
**`rspline` convergence check is dead code** — Low · `iter_spline.py:30-33`

The second `outliers = ...` is computed and never used, and `np.ma.where(mask_outliers)` is applied to a tuple of index arrays (so index 0 counts as "false"). Result: all `PSPLINES_MAXITER` (10) spline fits always run, observed 10 `LSQUnivariateSpline` calls on flat flux + noise, where the set of retained points is already constant (577 points) from the 5th fit on. Also, only *downward* outliers are clipped, so flares are not rejected (see BUG-028).
**Fix:** test the clip count of the current iteration. **Test:** `test_bug020_*`.

## BUG-021
**`flatten` docstring** — Low · issue #55

`mask`, `max_splines`, `stdev_cut`, `return_nsplines`, `verbose` are undocumented. Also: "Rime-windowed sliders" (typo), a stray backtick in "``rspline`", and `edge_cutoff : float, default: None` while the signature default is `0`. **Test:** `test_bug021_*` (checks every signature parameter is documented).

## BUG-022
**LOWESS tricube radius uses the second-to-last point** — Low · `lowess.py:10`

`radius = max(xi - x[0], x[-2] - xi)`. For a point at the left end of the data the window's *last* point lies beyond the radius, so its tricube weight is `(1 - r³)³` with r > 1 — non-zero (here positive after normalisation). Spike of +1 on the last member of the first 10-point window: wotan's trend at the first sample is 1.0754, `statsmodels.lowess` gives 1.0 exactly. Impact falls off ≈ k⁻³ with the window size, so it matters for short windows and edge samples (observed 6e-5 vs statsmodels with 150-point windows).
**Fix:** `x[-1]` (and consider the half-open window of BUG-002). **Test:** `test_bug022_*`.

## BUG-023
**`wotan/norms.py` runs demo code on import and uses a removed NumPy API** — Low

Lines 163-178 execute and print at import time; `np.alltrue` was removed in NumPy 2.0, so `import wotan.norms` raises `AttributeError`. The module is not used by `flatten` (dead code), but it breaks tooling that imports every module (coverage, docs). **Fix:** delete it or guard the demo and use `np.all`. **Test:** `test_bug023_*`.

## BUG-024
**`mask` is silently ignored by several methods** — Medium · `flatten.py`

`pspline`, `cofiam`, `supersmoother`, `savgol`, `medfilt` (and `ridge`/`lasso`/`elasticnet` by code reading) never receive the mask: output is bit-identical with and without it, and nothing warns. A user who masks a known transit believes it is protected. Related: open issues #31/#35 (extend masking to more methods).
**Fix:** support it, or at least `warnings.warn` / raise when `mask` is given for these methods.
**Test:** `test_bug024_*` (5). (`rspline` *does* honour the mask; the dip it removes anyway by clipping makes the two runs look similar.)

---

## BUG-025
**`flatten` divides by the trend: zero-crossing flux gives wild ratios without a warning** — Medium · `flatten.py` (formerly N-1) · issue #64

`flatten_lc = flux / trend`. For flux that crosses or approaches zero (mean-subtracted flux, magnitudes, ΔT) the ratio explodes; a trend of exactly 0 becomes NaN (documented in the source). The Huber spline itself is fine (`test_hspline_trend_is_not_forced_to_zero_issue_64`); the surprise is the division. Example: `0.01*sin(t)` with `biweight` returns values up to ~100 (together with BUG-003 dropping the negative samples), and nothing warns.
**Fix:** warn when the trend has non-positive or near-zero values, document it, and/or offer an additive mode (`flux - trend`). See also #56, #60. **Test:** `test_bug025_*`.

## BUG-026
**`cosine`/`cofiam` are not invariant to a time offset** — Low · `cofiam.py:matrix_gen` (formerly N-2) · issue #51

The basis uses the raw `t` (columns `1` and `t`, plus sin/cos of `t`), so at `t ≈ 2.457e6` (BJD) the least-squares problem is ill-conditioned. On a 1500-point, 30 d light curve `cosine` changes by up to 1.7e-4 when `t` is shifted by 2457000 d (sliders: exactly 0). In an earlier setup `lowess` changed by 1.7e-5; with the current test data it is 3e-9, so only `cosine`/`cofiam` are logged.
**Fix:** subtract `min(t)` inside `matrix_gen`/`detrend_cosine`. **Test:** `test_bug026_*`.

## BUG-027
**`hspline` precision depends on the flux offset** — Low · `huber_spline.py` (formerly N-3) · same family as BUG-009

The scikit-learn `HuberRegressor` tolerance is absolute. On a 1500-point, 30 d light curve with a 0.01 amplitude signal, adding +1000 to the flux changes the trend by 4.5e-4 (+5: 1.7e-7). Other configurations tried were fine (< 3e-7), so this only bites for large offsets with long light curves. Realistic normalised flux (≈1) is unaffected.
**Fix:** normalise the flux (divide by its median) inside the regression, or use a relative tolerance. **Test:** `test_bug027_*`.

## BUG-028
**Robust clipping is one-sided: flares are not rejected** — Medium · `cofiam.py` (`detrend_cosine`), `iter_spline.py` (formerly N-4)

`1 - detrended_flux > c*std` flags only points *below* the trend (transits) but the docstring says "2-sigma outliers". A 0.03 flare of 0.4 d changes the robust cosine trend by 4.8e-3 and the `rspline` trend by 6.4e-3, while the equally large *dip* barely moves them. Flares in the baseline therefore leave a bump in the trend and a negative dent in the flattened curve.
**Fix:** clip with `abs(1 - detrended) > c*std` (optionally with a separate upper threshold). **Test:** `test_bug028_*` (2).

## BUG-029
**An isolated one-sample segment silently gets a NaN trend** — Low · `slider.py` (formerly N-6) · *confidence: low*

With `break_tolerance` a segment of one sample has no neighbours and its trend is NaN, with no warning. (A 2-sample segment returns `trend[1] == flux[0]`, a consequence of BUG-002.) The intended behaviour is not documented: it could be the sample itself, or NaN plus a warning.
**Test:** `test_bug029_*`.

## BUG-030
**Bare `except:` clauses** — Low · `slider.py:98,147`, `iter_spline.py:26`, `flatten.py:259`, `gp.py:12`, `huber_spline.py:12`, `pspline.py:18`, `regression.py:12` (formerly N-7)

They also swallow `KeyboardInterrupt` and `SystemExit`. The `huber` fallback loop (`slider.py:147`) eats Ctrl-C once per window — possibly related to the "can't Ctrl-C" remark in issue #30 (which used `biweight`, so not the whole story).
**Fix:** `except Exception:`; for the Huber fallback catch the specific errors. **Test:** `test_bug030_*` (AST scan).

## BUG-031
**`inf` in the flux is not treated as invalid** — Medium · `flatten.py` (formerly N-8)

Only NaN is removed. With one `+inf` sample the `biweight` trend is non-finite for 29 of 300 samples (the NaN equivalent: 1), and `inf*0` under `fastmath=True` is not something to rely on. `median` happened to survive.
**Fix:** drop non-finite values (`np.isfinite`) instead of `isnan`. **Test:** `test_bug031_*`.

---

## Notes (not bugs)

* **N-5 — existing `wotan/tests.py`.** The repo does have a 519-line script of that name (run by `.travis.yml`), but it is not a pytest suite: it downloads a TESS file from `archive.stsci.edu`, needs astropy, uses hard-coded regression numbers that bake in the bugs above (e.g. the `slide_clip` sum, BUG-006), and CI pins Python 3.6/3.7 and scikit-learn 0.20, so BUG-001 was invisible. The new `tests/` suite is offline, deterministic (fixed seeds) and does not replace those TESS regression numbers.

Notes N-1…N-4 and N-6…N-8 of earlier drafts are now BUG-025…BUG-031.

## Open/closed upstream issues checked against the code

| Issue | Verdict | Where covered |
|---|---|---|
| #6, #19 huber non-convergence | Handled by design (median fallback); bare `except` noted (BUG-030) | `test_huber.py::test_falls_back_to_the_median…` |
| #18 andrewsinewave ≡ biweight | Fixed: the estimators differ | `test_location_estimates.py` |
| #30 memory growth | Not reproduced (numba 0.68: 0 MB growth over 60 calls) | `test_flatten_api.py::test_memory_does_not_grow…` |
| #31 / #35 masking | Works for sliders, lowess, cosine, gp, hspline, rspline; **silently ignored elsewhere → BUG-024** | `test_slider.py`, `test_known_bugs.py` |
| #51 BJD offset | Fixed for sliders/splines; cosine still affected (BUG-026) | `test_slider.py`, `test_cosine.py`, `test_lowess.py` |
| #54 finance example NaN | Plausibly the `(n, 1)` array → **BUG-018** (could not run the user's data) | `test_known_bugs.py` |
| #55 `mask` undocumented | Confirmed → **BUG-021** | `test_known_bugs.py` |
| #56 zeros → NaN; vectorise | Zero *trend* → NaN is intended (BUG-025); non-positive flux → **BUG-003** | `test_flatten_api.py` |
| #57 pspline small gaps | Fixed (regression test passes) | `test_splines.py` |
| #61 numba/numpy dtype error | Not reproduced (robust cosine + mask runs) | `test_cosine.py` |
| #64 huber spline zero-crossing | Not an hspline bug; division in `flatten` (BUG-025) | `test_splines.py` |
| #66 small cadence removes transits | User error (time not in days); window-vs-T14 behaviour matches the paper | `test_biweight.py` |
| #50, #53, #33, #62, #65, #67 | Usage / dependency / feature questions; not testable | — |

## Not verified

* The TESS regression numbers in `wotan/tests.py` (no network access in my workspace).
* Numerical values of the GP kernels beyond "tracks the trend" (scikit-learn optimiser dependent).
* The `supersmoother` package's own algorithm (only wotan's wrapper).
* Performance claims (README speed statements, issue #56 benchmark).
