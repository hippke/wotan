Installation
=====================================

To install the released version, type

    $ pip install wotan

which automatically installs `numpy`, `numba` and `scipy` if not present. Depending on the algorithm, additional dependencies exist:

- `lowess` and `huber` depend on `statsmodels`
- `hspline` and `gp` depend on `sklearn`
- `pspline` depends on `pygam`
- `supersmoother` depends on `supersmoother`

To install all additional dependencies, type ``$ pip install statsmodels sklearn supersmoother pygam``.

A known incompatibility exists between versions 1.3 of `scipy` and 0.9 of `statsmodels`, as the latest version of scipy deprecated the import for `factorial` from `scipy.misc`. This should be fixed again in a future version of `statsmodels`. Until then, I recommend to `pip install scipy==1.2` (or `conda install scipy==1.2`, if you use conda.).