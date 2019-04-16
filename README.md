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
   - ``biweight`` [Tukey's biweight](https://books.google.de/books?id=pGlHAAAAMAAJ)
   - ``andrewsinewave`` [Andrew's sine wave](http://www.jstor.org/stable/j.ctt13x12sw.3)
   - ``hodges`` [Hodges-Lehmann](https://doi.org/10.1214/aoms/1177704172)-[Sen](https://doi.org/10.2307/2527532)
   - ``welsch`` [Welsch](https://doi.org/10.1080/03610917808812083)-[Leclerc](https://doi.org/10.1007/BF00054839)
   - ``median`` The most robust (but least efficient)
   - ``medfilt`` Is a cadence-based median filter (*not* time-windowed)
   - ``mean`` The least robust (but most efficient for white noise)
   - ``trim_mean`` A trimmed mean with adjustable caps
- Splines, polynomials, and others:
   - ``lowess`` Locally weighted/estimated scatterplot smoothing ([Cleveland 1979](https://doi.org/10.1080/01621459.1979.10481038))
   - ``untrendy`` Spline with least-squares iteratively sigma-clipping re-weighting ([based on this package](https://github.com/dfm/untrendy))
   - ``cofiam`` Cosine Filtering with Autocorrelation Minimization ([Kipping et al. 2013](http://adsabs.harvard.edu/abs/2013ApJ...770..101K))
   - ``huberspline`` Univariate B-splines with a robust Huber estimator ([Huber 1981](https://books.google.de/books?id=hVbhlwEACAAJ))
   - ``savgol`` Sliding segments are fit with polynomials ([Savitzky & Golay 1964](https://ui.adsabs.harvard.edu/#abs/1964AnaCh..36.1627S)), cadence-based
- ``supersmoother`` [Friedman's (1984)](https://www.slac.stanford.edu/pubs/slacpubs/3250/slac-pub-3477.pdf) Super-Smoother, a local linear regression with adaptive bandwidth
- Gaussian Processes
   - ``gp_sqaredexp`` Squared-exponential kernel
   - ``gp_matern`` Matern 3/2 kernel
   - ``gp_periodic`` Periodic kernel informed by a Lomb-Scargle periodogram pre-search


Available features
-------------------

- ``window_length`` The length of the filter window in units of ``time`` (usually days).
- ``break_tolerance`` If there are large gaps in time, especially with corresponding flux level offsets, the detrending is much improved when splitting the data into several sub-lightcurves and applying the filter to each individually. Comes with an empirical default and is fully adjustable.
- ``edge_cutoff`` Trends near edges are less robust. Depending on the data, it may be beneficial to remove edges.
- ``cval`` Tuning parameter for the robust estimators (see [documentation](https://wotan.readthedocs.io/en/latest/index.html))
- ``ftol`` Desired precision of the final location estimate, using Newton-Raphson iteration. Default: `1e-6` (1 ppm).
- ``return_trend`` If `True`, the method will return a tuple of two elements (``flattened_flux``, ``trend_flux``) where ``trend_flux`` is the removed trend. Otherwise, it will only return ``flattened_flux``.


Installation
------------
To install the released version, type

    $ pip install wotan

Wotan requires numpy and numba to run. Additional dependencies:

- `lowess` requires `statsmodels`
- `huberspline` requires `sklearn` and `scipy`
- `supersmoother` requires `supersmoother`
- `CoFiAM` and `medfilt` require `scipy`

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
