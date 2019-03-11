import numpy
import batman
import matplotlib.pyplot as plt
from wotan import trend


if __name__ == "__main__":
    numpy.random.seed(seed=0)  # reproducibility
    # Create test data
    start = -2
    days = 4
    samples_per_day = 48 *5 #*30  # 48
    samples = int(days * samples_per_day)  # 48
    t = numpy.linspace(start, start + days, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = 0
    ma.per = 365.25  # orbital period
    ma.rp = 6371 / 696342  # 6371 planet radius (in units of stellar radii)
    ma.a = 217  # semi-major axis (in units of stellar radii)
    ma.inc = 90  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.5]  # limb darkening coefficients
    ma.limb_dark = "linear"  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    y = m.light_curve(ma)  # calculates light curve

    idx = numpy.argmax(y<1)
    T14 = 2 * abs(t[idx])
    print('T14 (days)', T14)
    ppm = 5
    stdev = 10 ** -6 * ppm

    trials = 10
    steps = 25
    for trial in range(trials):
        
        noise = numpy.random.normal(0, stdev, int(samples))
        test_y = y + noise

        i = []
        j = []
        windows = numpy.linspace(T14/2, 2.5*T14, steps)
        for window in windows:
            trend1 = trend(t, test_y, method='biweight_iter', window=window, c=5)
            trend1[numpy.isnan(trend1)]=1
            dip1 = numpy.sum(numpy.ones(len(test_y))) - numpy.sum(test_y)
            dip2 = numpy.sum(numpy.ones(len(test_y))) - numpy.sum(trend1)
            ratio_preserved = (1 - (dip2 / dip1)) #* 1.05
            print(window / T14, ratio_preserved)
            i.append(window / T14)
            j.append(ratio_preserved)
        plt.plot(trial, trials, i, j, color='black', alpha=0.3)

    # Create noise and merge with flux
    #ppm = 5
    #stdev = 10 ** -6 * ppm
    #noise = numpy.random.normal(0, stdev, int(samples))
    #y = original_flux + noise

    #plt.plot(t, y, color='blue')
    #plt.plot(t, trend1, color='black')
    #plt.xlim(-0.5, 0.5)
    #plt.ylim(0.999875, 1.00001)
    #plt.savefig("loss.pdf")

    plt.grid(axis='x')
    plt.plot((0, 2.5), (1, 1), color='black', linewidth='0.75')
    plt.xlabel('w/T14')
    plt.ylabel('Fraction of flux preserved')
    plt.ylim(0, 1.1)
    plt.xlim(0.5, 2.5)
    #plt.xlim(0)
    #plt.plot(t, trend1, color='black')
    #plt.xlim(-0.5, 0.5)
    #plt.ylim(0.999875, 1.00001)
    plt.savefig("loss.pdf")

    # on average: 98.3% recovery, stabw 1%
