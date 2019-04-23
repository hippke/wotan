Wotan Interface
================


Detrending with the ``flatten`` module 
--------------------------------------

.. automodule:: flatten.flatten



Choosing the right window size
--------------------------------------

Shorter windows (or knot distances, smaller kernels...) remove stellar variability more effectively, but suffer a larger risk of removing the desired signal (the transit) as well. What is the right window size?

For the time-windowed sliders, the window should be 2-3 times longer than the transit duration (for details, read [the paper](www). The transit duration is

:math:`T_{14,{\rm max}} = (R_{\rm s}+R_{\rm p}) \left( \frac{4P}{\pi G M_{\rm s}} \right)^{1/3}`

for a central transit on a circular orbit. If you have a prior on the stellar mass and radius, and a (perhaps maximum) planetary period, ``wotan`` offers a convenience function to calculate :math:`T_{14,{\rm max}}`:

.. automodule:: t14.t14

As an example, we can calculate the duration of an Earth-Sun transit:

::

    from wotan import t14
    tdur = t14(R_s=1, M_s=1, P=365, small_planet=True)
    print(tdur)

This should print ~0.54 (days), or about 13 hours. To protect a transit that long, it is reasonable to choose a window size of 3x as long, or about 1.62 days. With the ``biweight`` time-windowed slider, we would detrend with these settings:

::

    from wotan import t14, flatten
    tdur = t14(R_s=1, M_s=1, P=365, small_planet=True)
    flatten_lc = flatten(time, flux, window_length=3 * tdur)

