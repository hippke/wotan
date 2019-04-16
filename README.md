![Logo](https://raw.githubusercontent.com/hippke/wotan/master/logo.png)

[![Documentation](https://img.shields.io/badge/documentation-%E2%9C%93-blue.svg)](https://wotan.readthedocs.io/en/latest/index.html)
[![Build Status](https://travis-ci.com/hippke/wotan.svg?branch=master)](https://travis-ci.com/hippke/wotan)

Wōtan...
====================

...is a free and open source algorithm to automagically remove stellar trends from light curves for exoplanet transit detection.

> In Germanic mythology, Odin (/ˈoːðinː/ Old High German: Wōtan) is a widely revered god. He gave one of his eyes to Mimir in return for wisdom. Thus, in order to achieve a goal, one sometimes has to turn a blind eye. In Richard Wagner's "Der Ring des Nibelungen", Wotan is the King of the Gods (god of light, air, and wind) and a bass-baritone. According to Wagner, he is the "pinnacle of intelligence".

Example usage
-------------
```
from wotan import flatten
flatten_lc, trend_lc = flatten(time, flux, window_length=0.5, method='biweight')
```

For more details, have a look at the [documentation](https://wotan.readthedocs.io) and [tutorials](https://github.com/hippke/wotan/tree/master/tutorials).

Available detrending algorithms
---------------------------------

- Time-windowed sliders with robust location estimates:
   - ``biweight``: [Tukey's biweight](https://books.google.de/books?id=pGlHAAAAMAAJ)
   - ``andrewsinewave``: [Andrew's sine wave](http://www.jstor.org/stable/j.ctt13x12sw.3)
   - ``hodges`` Hodges-Lehmann-Sen
   - ``welsch`` Welsch-Leclerc
   - ``median`` (the most robust)
   - ``mean`` (not robust)
   - ``trim_mean``
- Splines, polynomials, and others:
   - ``lowess`` Locally weighted/estimated scatterplot smoothing ([Cleveland 1979](https://doi.org/10.1080/01621459.1979.10481038))
   - Untrendy
   - Cosine Filtering with Autocorrelation Minimization (``cofiam``), 
   - Univariate B-splines with a robust Huber estimator (``huberspline``)
- Cadence-based sliders:
   - Median
   - Mean
   - Savitzky-Golay
- Friedman's `Supersmoother` (``supersmoother``)
- Gaussian Processes
   - Squares-exponential kernel
   - Matern 3/2 kernel
   - Periodic kernel


Available features
-------------------


Installation
------------
To install the released version, type

    $ pip install wotan

Wotan requires numpy and numba to run. Additional dependencies:

- `lowess` requires `statsmodels`
- `huberspline` requires `sklearn` and `scipy`
- `supersmoother` requires `supersmoother`
- `CoFiAM` requires `scipy`

To install all dependencies, type ``$ pip install numpy numba scipy statsmodels sklearn supersmoother``

Authors
-------
``wotan`` was created by [Michael Hippke](www.hippke.org)

Attribution
----------------
Please cite [Hippke et al. (2019, XXX)](https://XXX) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@ARTICLE{XXX,
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
