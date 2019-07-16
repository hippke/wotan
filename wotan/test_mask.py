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
path = 'https://archive.stsci.edu/hlsps/tess-data-alerts/'
filename = 'hlsp_tess-data-alerts_tess_phot_00207081058-s01_tess_v1_lc.fits'
time, flux = load_file(path + filename)

from wotan import flatten, transit_mask
flatten_lc1, trend_lc1 = flatten(
    time,
    flux,
    method='lowess',
    window_length=0.4,
    return_trend=True,
    robust=True,
    )

import matplotlib.pyplot as plt
plt.scatter(time, flux, s=3, color='black')
plt.plot(time, trend_lc1, color='blue', linewidth=2)
plt.show()
plt.close()
plt.scatter(time, flatten_lc1, s=3, color='black')
plt.show()
plt.close()
"""
from transitleastsquares import transitleastsquares
model = transitleastsquares(time, flatten_lc1)
results = model.power(period_min=10, n_transits_min=1)
print('Period', format(results.period, '.5f'), 'd')
print('Duration (days)', format(results.duration, '.5f'))
print('T0', results.T0)
"""
mask = transit_mask(
    time=time,
    period=14.77338,
    duration=0.21060,
    T0=1336.141095
    )
"""
mask = transit_mask(
    t=time,
    period=results.period,
    duration=2*results.duration,
    T0=results.T0)
print(mask)
"""
plt.scatter(time[~mask], flux[~mask], s=3, color='black')
plt.scatter(time[mask], flux[mask], s=3, color='orange')
plt.show()
plt.close()

flatten_lc1, trend_lc1 = flatten(
    time,
    flux,
    method='cosine',
    window_length=0.4,
    return_trend=True,
    robust=True,
    mask=mask
    )

flatten_lc2, trend_lc2 = flatten(
    time,
    flux,
    method='lowess',
    window_length=0.4,
    return_trend=True,
    robust=True,
    mask=mask
    )

plt.scatter(time, flux, s=1, color='black')
plt.plot(time, trend_lc1, color='blue', linewidth=2)
plt.plot(time, trend_lc2, color='red', linewidth=2, linestyle='dashed')
plt.show()
plt.close()

plt.scatter(time, flatten_lc1, s=1, color='black')
plt.show()

plt.scatter(time, flatten_lc2, s=1, color='black')
plt.show()

