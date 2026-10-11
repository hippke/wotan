from __future__ import print_function, division
import warnings
import numpy as np
import scipy.interpolate


def detrend_huber_spline(time, flux, mask, knot_distance, cval=1.35):
    """Robust B-Spline regression with scikit-learn"""
    try:
        from sklearn.base import TransformerMixin
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import HuberRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError('Could not import sklearn')

    class BSplineFeatures(TransformerMixin):
        def __init__(self, knots):
            self.bsplines = self.get_bspline_basis(knots)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            features = np.zeros((len(X), len(self.bsplines)))
            for ispline, spline in enumerate(self.bsplines):
                features[:, ispline:ispline+1] = scipy.interpolate.splev(X, spline)
            return features

        def get_bspline_basis(self, knots):
            knots, coeffs, degree = scipy.interpolate.splrep(knots, np.zeros(len(knots)))
            bsplines = []
            for ispline in range(len(knots)):
                coeffs = [1 if ispl == ispline else 0 for ispl in range(len(coeffs))]
                bsplines.append((knots, coeffs, degree))
            return bsplines

    masked_flux = flux[mask==1]
    masked_time = time[mask==1]

    if len(masked_time) == 0:
        return np.full(len(time), np.nan)
    else:
        duration = np.max(masked_time) - np.min(masked_time)
        no_knots = int(duration / knot_distance)
        knots = np.linspace(np.min(masked_time), np.max(masked_time), no_knots)

    if len(knots) < 4:
        warnings.warn(
            "hspline: window_length (%g) is too large for a segment of length %g "
            "(fewer than 4 knots). The trend of this segment is NaN. Use a window "
            "shorter than a quarter of the segment." % (knot_distance, duration),
            UserWarning)
        return np.full(len(time), np.nan)
    else:
        # Standardise the flux: the optimiser tolerance is absolute, so the fit must
        # not depend on the flux level or unit
        offset = np.median(masked_flux)
        scale = 1.4826 * np.median(np.abs(masked_flux - offset))
        if not scale > 0:
            scale = 1.0
        # StandardScaler on the spline features and max_iter=1000 improve convergence;
        # ``cval`` is the epsilon of the HuberRegressor (default 1.35)
        pipeline = make_pipeline(
            BSplineFeatures(knots), StandardScaler(), HuberRegressor(max_iter=1000, epsilon=cval)
        )
        trend = pipeline.fit(masked_time[:, np.newaxis], (masked_flux - offset) / scale)
        return trend.predict(time[:, None]) * scale + offset
