![Logo](https://raw.githubusercontent.com/hippke/wotan/master/logo.png)

[![Documentation](https://img.shields.io/badge/documentation-%E2%9C%93-blue.svg)](https://wotan.readthedocs.io/en/latest/index.html)
[![Build Status](https://travis-ci.com/hippke/wotan.svg?branch=master)](https://travis-ci.com/hippke/wotan)

Wōtan...
====================

...offers free and open source algorithms to automagically remove stellar trends from light curves for exoplanet transit detection.

> In Germanic mythology, Odin (/ˈoːðinː/ Old High German: Wōtan) is a widely revered god. He gave one of his eyes to Mimir in return for wisdom. Thus, in order to achieve a goal, one sometimes has to turn a blind eye. In Richard Wagner's "Der Ring des Nibelungen", Wotan is the King of the Gods (god of light, air, and wind) and a bass-baritone. According to Wagner, he is the "pinnacle of intelligence".

Example usage
-------------
```
from wotan import flatten
flatten_lc, trend_lc = flatten(time, flux, window_length=0.5, method='biweight')
```

For more details, have a look at the [interactive playground](www), the [documentation](https://wotan.readthedocs.io) and [tutorials](https://github.com/hippke/wotan/tree/master/tutorials).

Available detrending algorithms
---------------------------------

- Time-windowed sliders with robust location estimates:
   - ``biweight`` [Tukey's biweight](https://books.google.de/books?id=pGlHAAAAMAAJ)
   - ``andrewsinewave`` [Andrew's sine wave](http://www.jstor.org/stable/j.ctt13x12sw.3)
   - ``hodges`` [Hodges-Lehmann](https://doi.org/10.1214/aoms/1177704172)-[Sen](https://doi.org/10.2307/2527532)
   - ``welsch`` [Welsch](https://doi.org/10.1080/03610917808812083)-[Leclerc](https://doi.org/10.1007/BF00054839)
   - ``huber`` [Huber's M-estimator (1981)](https://books.google.de/books/about/Robust_Statistics.html?id=hVbhlwEACAAJ&redir_esc=y)
   - ``median`` the most robust (but least efficient)
   - ``medfilt`` a cadence-based median filter (*not* time-windowed) for comparison
   - ``mean`` the least robust (but most efficient for white noise)
   - ``trim_mean`` trimmed mean with adjustable caps
   - ``winsorize`` outliers are [*winsorized*](https://en.wikipedia.org/wiki/Winsorizing) to a specified percentile
- Splines:
   - ``untrendy`` Spline with iteratively sigma-clipping ([based on the "untrendy" package](https://github.com/dfm/untrendy))
   - ``hspline`` Spline with a robust Huber estimator ([Huber 1981](https://books.google.de/books?id=hVbhlwEACAAJ))
   - ``pspline`` Penalized spline to automatically select the knot distance [(Eilers 1996)](https://pdfs.semanticscholar.org/5e3d/4cf7824be321af95ac098595957d8a87bf68.pdf), with iteratively sigma-clipping
- Polynomials and others:
   - ``lowess`` Locally weighted scatterplot smoothing ([Cleveland 1979](https://doi.org/10.1080/01621459.1979.10481038))
   - ``cofiam`` Cosine Filtering with Autocorrelation Minimization ([Kipping et al. 2013](http://adsabs.harvard.edu/abs/2013ApJ...770..101K))
   - ``savgol`` sliding segments are fit with polynomials ([Savitzky & Golay 1964](https://ui.adsabs.harvard.edu/#abs/1964AnaCh..36.1627S)), cadence-based
   - ``supersmoother`` [Friedman's (1984)](https://www.slac.stanford.edu/pubs/slacpubs/3250/slac-pub-3477.pdf) Super-Smoother, a local linear regression with adaptive bandwidth
- ``gp`` Gaussian Processes offering:
   - ``squared_exp`` Squared-exponential kernel
   - ``matern`` Matern 3/2 kernel
   - ``periodic`` Periodic kernel informed by a user-specified period
   - ``periodic_auto`` Periodic kernel informed by a Lomb-Scargle periodogram pre-search


Available features
-------------------

- ``window_length`` The length of the filter window in units of ``time`` (usually days).
- ``break_tolerance`` If there are large gaps in time, especially with corresponding flux level offsets, the detrending is much improved when splitting the data into several sub-lightcurves and applying the filter to each individually. Comes with an empirical default and is fully adjustable.
- ``edge_cutoff`` Trends near edges are less robust. Depending on the data, it may be beneficial to remove edges.
- ``cval`` Tuning parameter for the robust estimators (see [documentation](https://wotan.readthedocs.io/en/latest/index.html))
- ``return_trend`` If `True`, the method will return a tuple of two elements (``flattened_flux``, ``trend_flux``) where ``trend_flux`` is the removed trend. Otherwise, it will only return ``flattened_flux``.


Installation
------------
To install the released version, type

    $ pip install wotan

which automatically installs `numpy` and `numba` if not present. Depending on the algorithm, additional dependencies exist:

| Method          |`scipy`   |`sklearn`   |`statsmodels`   |`supersmoother`   | `untrendy`       | `pygam`       |
| ----------------|:--------:|:----------:|:--------------:|:----------------:|:----------------:|:----------------:|
| `medfilt`       | ✓        |            |                |                  |                  |     |
| `CoFiAM`        | ✓        |            |                |                  |                  |     |
| `hspline`       | ✓        | ✓          |                |                  |                  |     |
| `gp`            | ✓        | ✓          |                |                  |                  |     |
| `lowess`        |          |            | ✓              |                  |                  |     |
| `huber `        |          |            | ✓              |                  |                  |     |
| `supersmoother` |          |            |                | ✓                |                  |     |
| `untrendy`      |          |            |                |                  | ✓                |     |
| `pspline`      |          |            |                |                  | ✓                |  ✓   |


To install all dependencies, type ``$ pip install numpy numba scipy statsmodels sklearn supersmoother untrendy pygam``.

Attribution
----------------
Please cite [Hippke et al. (2019, XXX)](https://XXX) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@ARTICLE{XXX,
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Originality
----------------

As all scientific work, wōtan is [*standing on the shoulders of giants*](https://en.wikiquote.org/wiki/Isaac_Newton). Particularly, many detrending methods are wrapped from existing packages. Original contributions include:
- A time-windowed detrending master module with edge treatments and segmentation options
- Robust location estimates using Newton-Raphson iteration to a precision threshold for Tukey's biweight, Andrew's sine wave, and the Welsch-Leclerc. This is probably a "first", which reduces jitter in the location estimate by ~10 ppm
- Robustified penalized splines for automatic knot distance determination and outlier resistance
- Bringing together many methods in one place in a common interface, with sensible defaults
- Providing documentation, tutorials, and a [paper](www) which compares and benchmarks the methods
