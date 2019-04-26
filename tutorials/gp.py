import numpy as np
from astropy.io import fits
from transitleastsquares import resample

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
#filename = 'P:/P/Dok/tess_alarm/hlsp_tess-data-alerts_tess_phot_00077031414-s02_tess_v1_lc.fits'
#filename = 'tess2018206045859-s0001-0000000201248411-111-s_llc.fits'
time, flux = load_file(filename)
time, flux = resample(time, flux, factor=5)

# Use wotan to detrend
from wotan import flatten
"""
flatten_lc1, trend_lc1 = flatten(time, flux, kernel_size=0.25, return_trend=True, method='gp', kernel='matern')
flatten_lc2, trend_lc2 = flatten(time, flux, kernel_size=5, return_trend=True, method='gp', kernel='squared_exp')

# You see that we have to choose a different ``kernel_size`` for each method to 
# get similar results with respect to underfitting/overfitting

"""
# Plot the result
import matplotlib.pyplot as plt
"""
plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, linewidth=2, color='red')
plt.plot(time, trend_lc2, linewidth=2, color='blue', linestyle='dashed')

plt.xlim(1339.75, 1348)
plt.show()
"""
"""
# For (strictly) periodic signals, we can use ``kernel=periodic_auto`` which detects
# the strongest periodic signal using a Lomb-Scargle periodogram search, and then
# fits a GP with a periodic (ExpSineSquared) kernel. 
points = 1000
time = np.linspace(0, 15, points)
flux = 1 + np.sin(time)  / points
noise = np.random.normal(0, 0.0001, points)
flux += noise

for i in range(points):  
    if i % 75 == 0:
        flux[i:i+5] -= 0.0004  # Add some transits
        flux[i+50:i+52] += 0.0002  # and flares

#flux_synth = flux_synth * 1.001 * time_synth
#flux_synth[50:100] = np.nan
flatten_lc, trend_lc = flatten(
    time,
    flux,
    method='gp',
    kernel='periodic_auto',
    kernel_size=5,
    return_trend=True)

#plt.scatter(time, flux_synth, s=1, color='black')
#plt.plot(time, trend_lc, color='red', linewidth=2)

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc, color='red', linewidth=2)


plt.show()
plt.close()
plt.scatter(time, flatten_lc, s=1, color='black')
plt.show()
"""

# It also often works well for data that is periodic with additional trends and gaps:

points = 1000
time = np.linspace(0, 15, points)
flux = 1 + ((np.sin(time) +  + time / 10 + time**1.5 / 100) / 1000)
noise = np.random.normal(0, 0.0001, points)
flux += noise

for i in range(points):  
    if i % 75 == 0:
        flux[i:i+5] -= 0.0004  # Add some transits
        flux[i+50:i+52] += 0.0002  # and flares
flux[300:400] = np.nan

flatten_lc, trend_lc = flatten(
    time,
    flux,
    method='gp',
    kernel='periodic_auto',
    kernel_size=5,
    return_trend=True)

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc, color='red', linewidth=2)


plt.show()
plt.close()
plt.scatter(time, flatten_lc, s=1, color='black')
plt.show()
