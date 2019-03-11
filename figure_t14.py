import numpy
import matplotlib.pyplot as plt
from numpy import pi


def T14(R_s, M_s, P):
    """Input:  Stellar radius and mass; planetary period
                   Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""

    G = 6.673e-11  # gravitational constant [m^3 / kg / s^2]
    R_sun = 695508000  # radius of the Sun [m]
    R_jup = 69911000  # radius of Jupiter [m]
    M_sun = 1.989 * 10 ** 30  # mass of the Sun [kg]
    SECONDS_PER_DAY = 86400

    P = P * SECONDS_PER_DAY
    R_s = R_sun * R_s
    M_s = M_sun * M_s
    T14max = (R_s + 2 * R_jup) * ((4 * P) / (pi * G * M_s)) ** (1 / 3)  # planet 2 R_jup
    return T14max / SECONDS_PER_DAY

print(T14(R_s=1, M_s=1, P=0.31))

periods = numpy.geomspace(0.1, 1000, 1000)

plt.plot(periods, T14(R_s=1, M_s=1, P=periods), color='black')
plt.plot(periods, T14(R_s=0.13, M_s=0.1, P=periods), color='black')
plt.plot(periods, T14(R_s=1.7, M_s=2.1, P=periods), color='black')
plt.text(400, 0.35, 'M8')
plt.text(400, 0.62, 'G2')
plt.text(400, 0.78, 'A5')
plt.xlim(0, 1400/3)
plt.ylim(0, 1)
plt.xlabel('Planetary period  (days)')
plt.ylabel(r'Transit duration $T_{14}$ (days)')
#plt.xscale('log')
#plt.yscale('log')
plt.savefig('figure_t14.pdf')

