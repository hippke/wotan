# Generate synthetic data
import numpy as np
points = 1000
time = np.linspace(0, 15, points)
flux = 1 + ((np.sin(time) +  + time / 10 + time**1.5 / 100) / 1000)
noise = np.random.normal(0, 0.0001, points)
flux += noise
for i in range(points):  
    if i % 75 == 0:
        flux[i:i+5] -= 0.0004  # Add some transits
        flux[i+50:i+52] += 0.0002  # and flares
flux[400:500] = np.nan  # a data gap

# Use wotan to detrend
from wotan import flatten
flatten_lc, trend_lc = flatten(time, flux, window_length=0.5, return_trend=True)

# Plot the result
import matplotlib.pyplot as plt

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc, linewidth=2, color='red')
plt.show()

plt.scatter(time, flatten_lc, s=1, color='black')
plt.show()

# window_length balances between under-fit and over-fit
flatten_lc1, trend_lc1 = flatten(time, flux, window_length=0.2, return_trend=True)
flatten_lc2, trend_lc2 = flatten(time, flux, window_length=0.5, return_trend=True)
flatten_lc3, trend_lc3 = flatten(time, flux, window_length=1, return_trend=True)
plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='blue')  # over-fit
plt.plot(time, trend_lc2, linewidth=2, color='red')  # about right
plt.plot(time, trend_lc3, linewidth=2, color='orange')  # under-fit
plt.xlim(0, 2)
plt.ylim(0.9995, 1.0015)
plt.show()

# edge_cutoff removes edges
# Is the feature right at the start a signal that we want to keep?
# A visual examination is inconclusive
# For the purpose of a blind transit search, it is (slightly) preferable to remove edges

# Note that we set ``edge_cutoff=0.5``, but only 0.25 are removed - maximum half a window

flatten_lc1, trend_lc1 = flatten(time, flux, window_length=0.5, return_trend=True)
flatten_lc2, trend_lc2 = flatten(time, flux, window_length=0.5, edge_cutoff=0.5, return_trend=True)
plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='blue', linestyle='dashed')
plt.plot(time, trend_lc2, linewidth=2, color='red')
plt.xlim(0, 2)
plt.ylim(0.9995, 1.0015)
plt.show()

# break_tolerance
# If there are large gaps in time, especially with corresponding flux level offsets, 
# the detrending is much improved when splitting the data into several sub-lightcurves 
# and applying the filter to each individually. The default is window_length/2.

# break_tolerance=0 disables the breaks. Then, the whole dataset is one chunk
# Positive values, e.g., break_tolerance=0.1, split the data into chunks if there are
# breaks longer than 0.1 days (which is the case here)

# Usually, splines are much more sensitive to offets than slider-based detrenders.
# In this case, the spline is much improved if segments are used (break_tolerance>0)

points = 1000
time = np.linspace(0, 15, points)
flux = 1 + ((np.sin(time) +  + time / 10 + time**1.5 / 100) / 1000)
noise = np.random.normal(0, 0.00005, points)
flux += noise
for i in range(points):  
    if i % 75 == 0:
        flux[i:i+5] -= 0.0004  # Add some transits
        flux[i+50:i+52] += 0.0002  # and flares
flux[425:475] = np.nan  # a data gap
flux[475:] -= 0.002
flatten_lc1, trend_lc1 = flatten(time, flux, break_tolerance=0.1, window_length=1, method='hspline', return_trend=True)
flatten_lc2, trend_lc2 = flatten(time, flux, break_tolerance=0, window_length=1, method='hspline', return_trend=True)
plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc2, linewidth=2, color='red')
plt.plot(time, trend_lc1, linewidth=2, color='blue', linestyle='dashed')
plt.xlim(2, 11)
plt.ylim(0.9982, 1.0012)
plt.show()
