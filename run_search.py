import glob
import os
import numpy
import scipy.signal
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs
from astropy.stats import sigma_clip
from astropy.io import fits
from wotan_all import trend as trend_all
from transitleastsquares import (
    transitleastsquares,
    cleaned_array,
    catalog_info,
    transit_mask
    )
from vetting_sheet import vetting_figure



def load_file(filename):
    """Loads a TESS *spoc* FITS file and returns cleaned arrays TIME, PDCSAP_FLUX"""
    hdu = fits.open(filename)
    q = hdu[1].data['QUALITY'] == 0
    print(q)
    t = hdu[1].data['TIME']
    y = hdu[1].data['PSF_FLUX']  # values with non-zero quality are nan or zero'ed
    y = y[q]
    t = t[q]
    t, y = cleaned_array(t, y)  # remove invalid values such as nan, inf, non, negative
    print(
        'Time series from', format(min(t), '.1f'),
        'to', format(max(t), '.1f'),
        'with a duration of', format(max(t)-min(t), '.1f'), 'days')
    print('median cadence', format(get_cadence(t)*60*24, '.2f'), 'minutes')
    return t, y, hdu[0].header['TIC_ID']


def get_cadence(t):
    """Calculates the median time between steps in time series"""
    return numpy.median(numpy.diff(t))


if __name__ == "__main__":

    os.chdir("Z:/wotan/fits/")
    for file in glob.glob("*.fits"):
        try:
            print('Working on file', file)
            time, flux, TIC_ID = load_file(file)
            rawflux = flux.copy()
            rawtime = time.copy()
            print('TIC_ID', TIC_ID)

            tess_mag = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID).as_array()[0][60]
            print('TESS magnitude', tess_mag)

            # Remove stellar variability
            trend = trend_all(time, flux, method='huberspline', window=0.25)

            #plt.scatter(time, flux, s=1, color='black')
            #plt.plot(time, trend, color='red')
            #plt.show()
            #plt.close()

            y_filt = flux / trend
            y_filt = sigma_clip(y_filt, sigma_upper=3, sigma_lower=float('inf'))

            t_filt, y_filt = cleaned_array(time, y_filt)

            #plt.scatter(t_filt, y_filt, s=1, color='black')
            #plt.show()
            #plt.close()


            # Get mass, radius and limb darkening priors from TESS Input Catalog
            ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=TIC_ID)
            print('Searching with limb-darkening estimates using quadratic LD (a,b)=', ab)

            # Perform the search
            model = transitleastsquares(t_filt, y_filt)
            results = model.power(
                #n_transits_min=1,  # If you like to search for single transit events
                u=ab,
                #period_min=1.7,
                #period_max=2
                #oversampling_factor=5,
                #duration_grid_step=1.05)
                )

            # First set of simple vetting criteria:
            # Disable for now
            valid = True

            """
            if results.distinct_transit_count==1 and results.transit_count >= 2:
                valid = False
                print('Vetting fail! results.distinct_transit_count==1 and results.transit_count == 2')
            if results.distinct_transit_count==2 and results.transit_count >= 3:
                valid = False
                print('Vetting fail! results.distinct_transit_count==2 and results.transit_count == 3')
            if results.SDE < 8 :
                valid = False
                print('Vetting fail! results.SDE < 8', results.SDE)
            if results.snr < 7:
                valid = False
                print('Vetting fail! results.snr < 7', results.snr)

            upper_transit_depths = results.transit_depths + results.transit_depths_uncertainties
            if results.transit_count == 2 and max(upper_transit_depths) > 1:
                valid = False
                print('Vetting fail! 2 transits, only 1 significant')

            upper_transit_depths = results.transit_depths + results.transit_depths_uncertainties
            if results.transit_count == 3 and max(upper_transit_depths) > 1:
                valid = False
                print('Vetting fail! 3 transits, not all 3 significant')

            upper_transit_depths = results.transit_depths + results.transit_depths_uncertainties
            if results.transit_count == 4 and max(upper_transit_depths) > 1:
                valid = False
                print('Vetting fail! 4 transits, not all 4 significant')

            if results.depth < 0.95:
                valid = False
                print('Vetting fail! Transit depth < 0.95', results.depth)

            #if results.SDE > 9:  # 9
                print('Signal detection efficiency (SDE):', format(results.SDE, '.1f'))
                print('SNR:', format(results.snr, '.1f'))
                print('Search completed')
            """

            if valid:
                print('Valid result, attempting figure...')
                vetting_figure(
                    TIC_ID=TIC_ID,
                    planet_number=1,
                    results=results,
                    t=t_filt,
                    y=y_filt,
                    y_filt=y_filt,
                    trend=trend,
                    rawtime=rawtime,
                    rawflux=rawflux)
                print('Figure made')
            else:
                print('No figure made, vetting failed!')
        except:
            print('Failed for', file)
