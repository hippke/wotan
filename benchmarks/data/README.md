# Benchmark data

* `kepler/` - Kepler-10 (KIC 11904151) long-cadence light curves, all 15 quarter files
  (Q0-Q14 as listed by MAST), PDCSAP_FLUX is used. Re-download with
  `benchmarks/data/download_kepler10.sh`.
* `white_1m_flux.npy` - 1e6 samples of white noise, `1 + N(0, 1e-3)`, seed 1
  (`numpy.random.default_rng(1)`); time is `arange(1e6) * 2 min`.
* `../ref/` - reference trends from the baseline (master at 64f77d7) for the
  correctness check (max |diff| <= 1e-6).
