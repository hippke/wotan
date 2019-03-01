import numpy
import batman
import os
import psutil
from transitleastsquares import transitleastsquares


if __name__ == "__main__":
    print("Starting test: synthetic...", end='')
    numpy.random.seed(seed=0)  # reproducibility
    start = 48
    days = 365.25 * 3
    samples_per_day = 12  # 48
    samples = int(days * samples_per_day)  # 48
    t = numpy.linspace(start, start + days, samples)
    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = (start + 20)
    ma.per = 365.25  # orbital period
    ma.rp = 6371 / 696342  # 6371 planet radius (in units of stellar radii)
    ma.a = 217  # semi-major axis (in units of stellar radii)
    ma.inc = 90  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.5]  # limb darkening coefficients
    ma.limb_dark = "linear"  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    original_flux = m.light_curve(ma)  # calculates light curve

    # Create noise and merge with flux
    ppm = 5
    stdev = 10 ** -6 * ppm
    noise = numpy.random.normal(0, stdev, int(samples))
    y = original_flux + noise

    for i in range(10):
        process = psutil.Process(os.getpid())
        print('RAM (MB)', process.memory_info().rss / 2**20)
        model = transitleastsquares(t, y)
        results = model.power(
            period_min=360,
            period_max=370,
            transit_depth_min=10 * 10 ** -6,
            oversampling_factor=5,
            duration_grid_step=1.02,
            show_progress_bar=False
        )
