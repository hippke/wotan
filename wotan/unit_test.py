from wotan import detrend
import numpy
import time
import matplotlib.pyplot as plt
#import everest
from astropy.stats import sigma_clip


if __name__ == "__main__":
    print("Starting test: wotan synthetic...")
    numpy.random.seed(seed=0)  # reproducibility

    days = 10
    samples = 1000
    window = 1
    stdev = 0.01
    t = numpy.linspace(0, days, samples)
    data = numpy.random.normal(0, stdev, int(samples))
    #print('1', len(data))

    # compile JIT numba
    print("Numba compilation...", end="")
    t1 = time.perf_counter()
    trend = detrend(t, data, window, c=6, ftol=1e-6, maxiter=15)
    t2 = time.perf_counter()
    print("{0:.3f}".format(t2-t1), 'seconds')
    #print('2', len(data))

    print('Detrending 1...', end="")
    #print('3', len(data))
    t1 = time.perf_counter()
    trend = detrend(t, data, window, c=6, ftol=1e-6, maxiter=15)
    t2 = time.perf_counter()
    print("{0:.3f}".format(t2-t1), 'seconds')
    #print('4', len(data))
    #print(len(trend))


    numpy.testing.assert_equal(len(trend), 1000)
    numpy.testing.assert_almost_equal(numpy.nanmax(trend), 0.001927581851475834)
    numpy.testing.assert_almost_equal(numpy.nanmin(trend), -0.0024286115432384527)
    numpy.testing.assert_almost_equal(trend[500], -0.0006178575452455013)

    # Test code in  "def biweight_location_iter" for "if mad == 0"
    # This was once a bug in scikit-learn
    print('Detrending 2...', end="")
    t1 = time.perf_counter()
    data2 = numpy.zeros(samples)
    trend = detrend(t, data2, window, c=6, ftol=1e-6, maxiter=15)
    numpy.testing.assert_almost_equal(numpy.nanmin(trend), 0)  # not all nan
    t2 = time.perf_counter()
    print("{0:.3f}".format(t2-t1), 'seconds')
