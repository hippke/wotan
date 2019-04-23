from __future__ import print_function, division
import numpy
import wotan.constants as constants



def t14(R_s, M_s, P, small_planet=False):
    """Planetary transit duration assuming a central transit on a circular orbit
    
    Parameters
    ----------
    R_s : float
        Stellar radius (in solar radii)
    M_s : float
        Stellar mass (in solar masses)
    P : float
        Planetary period (in days)
    small_planet : bool, default: `False`
        Small planet assumption if `True` (zero planetary radius). Otherwise, a 
        planetary radius of 2 Jupiter radii is assumed (maximum reasonable planet size)
    
    Returns
    -------
    t14 : float
        Planetary transit duration (in days)
    """
    if small_planet:
        planet_size = 0
    else:
        planet_size = 2 * constants.R_jup
    t14 = (
        (constants.R_sun * R_s + planet_size)
        * (
            (4 * P * constants.SECONDS_PER_DAY)
            / (numpy.pi * constants.G * constants.M_sun * M_s)
        )
        ** (1 / 3)
    ) / constants.SECONDS_PER_DAY
    return t14
