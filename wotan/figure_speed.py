import numpy
import matplotlib.pyplot as plt
from wotan import flatten
import time as ttime

if __name__ == "__main__":
    numpy.random.seed(seed=0)  # reproducibility
    samples = 1000
    window_length = 0.5
    days = 30
    stdev = 0.01
    time = numpy.linspace(0, days, samples)
    flux = numpy.random.normal(0, stdev, int(samples))

    print("Warmup...")
    t1 = ttime.perf_counter()
    y_filt, trend1 = flatten(
        time,
        flux,
        window_length=window_length,
        edge_cutoff=0,
        method='biweight',
        return_trend=True)
    t2 = ttime.perf_counter()
    print(t2-t1)

    #plt.rc('font',  family='serif', serif='Computer Modern')
    #plt.rc('text', usetex=True)
    plt.figure(figsize=(4, 4 / 1.5))

    samples_array = numpy.geomspace(1000, 5000, 25)
    curve_biweight = []
    curve_huberspline = []
    curve_hodges = []
    curve_trim_mean = []

    for samples in samples_array:
        time = numpy.linspace(0, days, int(samples))
        flux = numpy.random.normal(0, stdev, int(samples))

        t1 = ttime.perf_counter()
        y_filt, trend1 = flatten(
            time,
            flux,
            window_length=window_length,
            method='biweight',
            return_trend=True)
        t2 = ttime.perf_counter()
        print(samples, t2-t1)
        curve_biweight.append(t2-t1)

        t1 = ttime.perf_counter()
        y_filt, trend1 = flatten(
            time,
            flux,
            window_length=window_length,
            method='huberspline',
            return_trend=True)
        t2 = ttime.perf_counter()
        print(samples, t2-t1)
        curve_huberspline.append(t2-t1)

        t1 = ttime.perf_counter()
        y_filt, trend1 = flatten(
            time,
            flux,
            window_length=window_length,
            method='hodges',
            return_trend=True)
        t2 = ttime.perf_counter()
        print(samples, t2-t1)
        curve_hodges.append(t2-t1)

        t1 = ttime.perf_counter()
        y_filt, trend1 = flatten(
            time,
            flux,
            window_length=window_length,
            method='mean',
            return_trend=True)
        t2 = ttime.perf_counter()
        print(samples, t2-t1)
        curve_trim_mean.append(t2-t1)

    plt.plot(samples_array, curve_biweight, color='black', linewidth=1, label='biweight, welsch, andrew')
    plt.plot(samples_array, curve_huberspline, color='red', linewidth=1, label='huberspline')
    plt.plot(samples_array, curve_trim_mean, color='blue', linewidth=1, label='mean')
    plt.plot(samples_array, curve_hodges, color='orange', linewidth=1, label='hodges')
    plt.legend()
    plt.xlabel('Data points')
    plt.ylabel('Time (sec)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(samples_array), max(samples_array))
    plt.savefig('figure_speed.pdf', bbox_inches='tight')
