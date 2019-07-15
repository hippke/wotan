from __future__ import division, print_function
import numpy as np


def transit_mask(t, period, duration, T0):
    """Calculates in-transit mask of time array for a given planet ephemeris

    Parameters
    ----------
    time : array-like
        Time values
    period : float
        Period of the planet (in units of time)
    duration : float
        Transit duration of the planet (in units of time)
    T0 : float
        Mid-transit time  (in units of time)

    Returns
    -------
    mask : array-like
        Boolean array of True/False values, where in-transit points are True
    """
    
    half_period = 0.5 * period
    with np.errstate(invalid='ignore'):  # ignore NaN values
        return np.abs((t - T0 + half_period) % period - half_period) < 0.5 * duration


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays."""

    def isvalid(value):
        valid = False
        if value is not None:
            if not np.isnan(value):
                if value < np.inf:
                    valid = True
        return valid

    # Start with empty Python lists and convert to numpy arrays later (reason: speed)
    clean_t = []
    clean_y = []
    if dy is not None:
        clean_dy = []

    # Cleaning numpy arrays with both NaN and None values is not trivial, as the usual
    # mask/delete filters do not accept their simultanous ocurrence without warnings.
    # Instead, we iterate over the array once; this is not Pythonic but works reliably.
    for i in range(len(y)):

        # Case: t, y, dy
        if dy is not None:
            if isvalid(y[i]) and isvalid(t[i]) and isvalid(dy[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])
                clean_dy.append(dy[i])

        # Case: only t, y
        else:
            if isvalid(y[i]) and isvalid(t[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])

    clean_t = np.array(clean_t, dtype=float)
    clean_y = np.array(clean_y, dtype=float)

    if dy is None:
        return clean_t, clean_y
    else:
        clean_dy = np.array(clean_dy, dtype=float)
        return clean_t, clean_y, clean_dy
