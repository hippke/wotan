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
