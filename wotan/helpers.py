from __future__ import division, print_function
import numpy as np


def transit_mask(time, period, duration, T0):
    half_period = 0.5 * period
    with np.errstate(invalid='ignore'):  # ignore NaN values
        return np.abs((time - T0 + half_period) % period - half_period) < 0.5 * duration


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays: the points where all values are valid (not
    None, not NaN, not +inf, not masked)."""

    def as_float(values):
        # None -> NaN, masked entries -> NaN
        return np.ma.asarray(values).astype(float).filled(np.nan)

    arrays = [as_float(t), as_float(y)] + ([as_float(dy)] if dy is not None else [])
    valid = np.ones(len(arrays[1]), dtype=bool)
    for values in arrays:
        valid &= ~np.isnan(values) & (values < np.inf)
    cleaned = tuple(values[valid] for values in arrays)
    return cleaned
