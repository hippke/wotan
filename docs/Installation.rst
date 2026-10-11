Installation
=====================================

To install the released version, type

    $ pip install wotan

which automatically installs `numpy`, `numba` and `scipy` if not present. Depending on the algorithm, additional dependencies exist:

- `huber`, `ramsay` and `hampel` depend on `statsmodels`
- `hspline` and `gp` depend on `scikit-learn`
- `pspline` depends on `pygam`
- `supersmoother` depends on `supersmoother`

To install all additional dependencies, type ``$ pip install statsmodels scikit-learn supersmoother pygam``.

To run the test suite and the benchmarks from a clone of the repository:

    $ pip install -r tests/requirements-test.txt
    $ pytest
    $ python benchmarks/bench.py --method biweight

Note for developers: wotan's compiled functions are cached by ``numba`` (``__pycache__/*.nbi``, ``*.nbc``). The cache does not track changes across files: after editing a function that another cached function calls (e.g. ``location_iter`` used by ``running_segment``), the caller keeps the old version. Delete the cache files after such edits (``find wotan -name "*.nb?" -delete``). Installed releases are not affected.
