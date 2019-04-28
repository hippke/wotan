import numpy as np
from astropy.io import fits

def load_file(filename):
    """Loads a TESS *spoc* FITS file and returns TIME, PDCSAP_FLUX"""
    hdu = fits.open(filename)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']
    flux[flux == 0] = np.nan
    return time, flux


print('Loading TESS data from archive.stsci.edu...')
#filename = "https://archive.stsci.edu/hlsps/tess-data-alerts/" \
#"hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"

filename = "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
filename = "https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00077031414-s02_tess_v1_lc.fits"
#filename = 'tess2018206045859-s0001-0000000201248411-111-s_llc.fits'
time, flux = load_file(filename)
#time = time[:5000]
#flux = flux[:5000]

# Use wotan to detrend
from wotan import flatten
flatten_lc1, trend_lc1 = flatten(time, flux, window_length=0.75, return_trend=True, method='mean')
flatten_lc2, trend_lc2 = flatten(time, flux, window_length=0.75, return_trend=True, method='biweight')

# Plot the result
import matplotlib.pyplot as plt

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='red')
plt.plot(time, trend_lc2, linewidth=2, color='blue')

plt.xlim(min(time), 1365)
plt.show()

# Any longer, and it would be an underfit

# Zoom into the issues that the running mean has:

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='red')
plt.plot(time, trend_lc2, linewidth=2, color='blue')

plt.xlim(1358.5, 1360.2)
plt.ylim(6725, 6900)
plt.show()


# Different robust methods
flatten_lc1, trend_lc1 = flatten(time, flux, window_length=0.75, return_trend=True, method='median')
flatten_lc2, trend_lc2 = flatten(time, flux, window_length=0.75, return_trend=True, method='biweight')

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='red')
plt.plot(time, trend_lc2, linewidth=2, color='blue')
plt.xlim(1358.5, 1360.2)
plt.ylim(6725, 6900)
plt.show()

# This case illustrates that the most robust detrender, the median, is even more 
# appropriate near such deep, wide transits. It has a downside, however: increased jitter.
# This is even visually evident when zooming into a non-transit time:


flatten_lc1, trend_lc1 = flatten(time, flux, window_length=0.75, return_trend=True, method='median')
flatten_lc2, trend_lc2 = flatten(time, flux, window_length=0.75, return_trend=True, method='biweight')

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='red')
plt.plot(time, trend_lc2, linewidth=2, color='blue')
plt.xlim(1370.85, 1371.3)
plt.ylim(6750, 6795)  #   6725, 6900
plt.show()

# For the purpose of transit detection, it has shown optimal to use a robust method
# such as the biweight. The maximal robustness of the median is, in total, not beneficial
# as the additional noise outweights the additional robustness. 

# The differences between the other sliders are often very small:


flatten_lc1, trend_lc1 = flatten(time, flux, window_length=0.75, return_trend=True, method='andrewsinewave')
flatten_lc2, trend_lc2 = flatten(time, flux, window_length=0.75, return_trend=True, method='biweight')
flatten_lc3, trend_lc3 = flatten(time, flux, window_length=0.75, return_trend=True, method='welsch')
flatten_lc4, trend_lc4 = flatten(time, flux, window_length=0.75, return_trend=True, method='huber')

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='red', linestyle='dotted')
plt.plot(time, trend_lc2, linewidth=2, color='blue')
plt.plot(time, trend_lc3, linewidth=2, color='orange', linestyle='dashed')
plt.plot(time, trend_lc4, linewidth=2, color='pink')
#plt.xlim(1370.85, 1371.3)
#plt.ylim(6750, 6795)  #   6725, 6900
plt.show()
