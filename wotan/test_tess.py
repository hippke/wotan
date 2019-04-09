import numpy
import matplotlib.pyplot as plt
from astropy.io import fits
from wotan import flatten


def load_file(filename):
    """Loads a TESS *spoc* FITS file and returns TIME, PDCSAP_FLUX"""
    hdu = fits.open(filename)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']
    flux[flux == 0] = numpy.nan
    return time, flux


def main():
    filename = "https://archive.stsci.edu/hlsps/tess-data-alerts/" \
        "hlsp_tess-data-alerts_tess_phot_00062483237-s01_tess_v1_lc.fits"
    time, flux = load_file(filename)
    flatten_lc, trend_lc = flatten(
        time,
        flux,
        window_length=0.5,
        edge_cutoff=1,
        break_tolerance=0.1,
        return_trend=True,
        cval=5.0)

    plt.scatter(time, flux, s=1, color='black')
    plt.plot(time, trend_lc, color='red')
    plt.show()
    plt.close()

    plt.scatter(time, flatten_lc, s=1, color='black')
    plt.show()


if __name__ == '__main__':
    main()
