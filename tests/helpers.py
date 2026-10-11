"""Shared helpers for the wotan test suite.

Nothing in here imports wotan's detrenders: the references are independent
re-implementations (or scipy / statsmodels / astropy), so a test can not
pass merely because the code under test agrees with itself.
"""
import numpy as np


# --------------------------------------------------------------------------- data
def make_lc(n=1500, span=30.0, period=10.0, amp=0.01, noise=5e-4, seed=0, level=1.0):
    """Evenly sampled light curve: ``level * (1 + amp*sin(2 pi t / period)) + noise``.

    Returns ``(time, flux, true_trend)``.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, span, n)
    true = level * (1.0 + amp * np.sin(2.0 * np.pi * t / period))
    flux = true + rng.normal(0.0, noise * level, n)
    return t, flux, true


def sinc(x):
    return np.sinc(x)  # numpy's sinc is sin(pi x) / (pi x)


def sliding_mean_sine_bias(amp, period, window):
    """Peak bias of a *sliding mean* of ``amp*sin(2 pi t/period)``.

    The mean of a sine over a window of length w is attenuated by
    sinc(w/period); the peak error is therefore amp * (1 - sinc(w/period)).
    Used to set principled tolerances for "does the trend follow the signal".
    """
    return amp * (1.0 - sinc(window / period))


def interior(t, window):
    """Boolean mask of samples whose full window lies inside the data."""
    return (t >= t.min() + window / 2.0) & (t <= t.max() - window / 2.0)


def away_from_end(t, window):
    """Samples whose window does not reach the last sample (see BUG-002)."""
    return t + window / 2.0 < t.max()


# --------------------------------------------------------------- references
def window_reference(t, f, window, estimator, mask_keep=None):
    """Brute-force sliding-window estimate.

    For every sample use all points with ``|t_j - t_i| < window/2``
    (strict, so ties on window borders do not matter for random sampling),
    optionally restricted to ``mask_keep``. NaN if the window is empty.
    """
    out = np.full(len(t), np.nan)
    keep = np.ones(len(t), bool) if mask_keep is None else mask_keep
    for i in range(len(t)):
        sel = keep & (np.abs(t - t[i]) < window / 2.0)
        if sel.any():
            out[i] = estimator(np.ascontiguousarray(f[sel], dtype=float))
    return out


def m_estimator_reference(x, c, kind, tol=1e-13, maxiter=10000):
    """Independent iterative M-estimator of location, started at the median.

    ``kind``: 'biweight' (Tukey), 'welsch' (truncated), 'andrew' (sine wave).
    Scale is the raw MAD (no 1.4826), ``c`` is in units of MAD - exactly the
    convention documented for wotan's ``cval``.
    """
    x = np.asarray(x, float)
    center = np.median(x)
    mad = np.median(np.abs(x - center))
    if mad == 0:
        return center
    for _ in range(maxiter):
        u = (x - center) / (c * mad)
        if kind == "biweight":
            w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        elif kind == "welsch":
            w = np.where(np.abs(u) < 1, np.exp(-u ** 2 / 2), 0.0)
        elif kind == "andrew":
            with np.errstate(invalid="ignore", divide="ignore"):
                w = np.where(np.abs(u) < np.pi, np.where(u == 0, 1.0, np.sin(u) / u), 0.0)
        else:
            raise ValueError(kind)
        step = np.sum((x - center) * w) / np.sum(w)
        center += step
        if abs(step) < tol:
            break
    return center


def measured_depth(flat, in_transit):
    """Transit depth measured on a flattened light curve."""
    return np.nanmean(flat[~in_transit]) - np.nanmean(flat[in_transit])


def inject_box_transit(t, flux, period, duration, t0, depth):
    """Return (flux with box transit, boolean in-transit mask)."""
    from wotan import transit_mask

    intr = transit_mask(t, period, duration, t0)
    out = flux.copy()
    out[intr] -= depth
    return out, intr


# ------------------------------------------------------------ transit scenario
def transit_scenario(cadence_min=4.0, span=30.0, period=10.0, depth=0.01,
                     noise=3e-4, amp=0.0, trend_period=15.0, seed=21):
    """Light curve with a periodic box transit of a 1 R_sun / 1 M_sun star.

    Returns ``dict(t, flux, intransit, t14, depth, true_trend)``.
    The transit duration comes from ``wotan.t14`` (small planet), i.e. the
    quantity the wotan paper sizes the detrending window against
    (window ~ 3 x T14).
    """
    from wotan import t14, transit_mask

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, cadence_min / 1440.0)
    dur = t14(R_s=1.0, M_s=1.0, P=period, small_planet=True)
    true = 1.0 + amp * np.sin(2 * np.pi * t / trend_period)
    flux = true + rng.normal(0.0, noise, len(t))
    intr = transit_mask(t, period, dur, 5.0)
    flux = flux.copy()
    flux[intr] -= depth
    return dict(t=t, flux=flux, intransit=intr, t14=dur, depth=depth, true_trend=true)
