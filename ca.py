import numba
import numpy
from numpy import sin, cos, pi
from scipy.optimize import leastsq


@numba.jit(nopython=True)
def detrend_curve(t, fact, D_max, k_max):
    ph = 2 * pi * t / D_max
    detr = numpy.ones(len(t)) * fact[0]
    for k in range(1, k_max + 1):
        detr += fact[2 * k - 1] * sin(ph * k) + fact[2 * k] * cos(ph * k)
    return detr


def find_detrending_for_region(t, y, ferr, k_max, D_max):

    def fit_func(fact):
        return (y - detrend_curve(t, fact, D_max, k_max)) / ferr

    def out_func(blub):
        return detrend_curve(t, best_param, D_max, k_max)

    x0 = numpy.zeros(2 * k_max + 1)
    x0[0] = numpy.nanmean(y)
    best_param, param_fit_status = leastsq(fit_func, x0)
    result = detrend_curve(t, best_param, D_max, k_max)
    return out_func


def detrend_light_curve_cofiam(t, y, ferr, window):

    D_max = 2 * (numpy.max(t) - numpy.min(t))
    k_max = max(1, min(100, int(D_max / window)))
    auto_corr_min = 1.0
    k_min_a_c = None
    detrend_func = None
    dw_mask = numpy.array([True] * len(t))
    for k_m in range(1, k_max + 1):
        detrend_func_temp = find_detrending_for_region(t, y, ferr, k_m, D_max)
        detrender_no_transit = detrend_func_temp(t)
        dw_y = (y[dw_mask] / detrender_no_transit[dw_mask] - 1)
        s = numpy.sum((dw_y[1:] - dw_y[:-1]) ** 2) / (numpy.sum(dw_y ** 2))
        auto_c_temp = (s - 2) / 2
        print(k_m, k_max, auto_c_temp)
        if numpy.abs(auto_c_temp) < numpy.abs(auto_corr_min):
            auto_corr_min = auto_c_temp
            detrend_func = detrend_func_temp
    return detrend_func(t)



import numpy
import batman
import matplotlib.pyplot as plt
from cofiam import detrend_light_curve_cofiam
from astropy.stats import sigma_clip, LombScargle
from astropy.io import fits
from transitleastsquares import (
    transitleastsquares,
    cleaned_array,
    catalog_info,
    transit_mask
    )

def load_file(filename):
    """Loads a TESS *spoc* FITS file and returns cleaned arrays TIME, PDCSAP_FLUX"""
    print(filename)
    hdu = fits.open(filename)
    t = hdu[1].data['TIME']
    y = hdu[1].data['PDCSAP_FLUX']  # values with non-zero quality are nan or zero'ed 
    t, y = cleaned_array(t, y)  # remove invalid values such as nan, inf, non, negative
    print(
        'Time series from', format(min(t), '.1f'), 
        'to', format(max(t), '.1f'), 
        'with a duration of', format(max(t)-min(t), '.1f'), 'days')
    return t, y, hdu[0].header['TICID']  # filename[36:45]  "trappist"


path = 'P:/P/Dok/tess_alarm/'
#file = 'hlsp_tess-data-alerts_tess_phot_00009006668-s02_tess_v1_lc.fits'
#file = 'hlsp_tess-data-alerts_tess_phot_00002760710-s02_tess_v1_lc.fits'
file = 'hlsp_tess-data-alerts_tess_phot_00009725627-s02_tess_v1_lc.fits'
t, y, TIC_ID = load_file(path + file)
y = y / numpy.median(y)
y = sigma_clip(y, sigma_lower=20, sigma_upper=3)
t, y = cleaned_array(t, y)

trend=detrend_light_curve_cofiam(t,y,numpy.ones(len(t)), window=2)

plt.scatter(t, y, s=1, color='black')
plt.plot(t, trend, color='red')
plt.show()

