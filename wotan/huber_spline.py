from __future__ import print_function, division
import numpy as np
import scipy.interpolate


def detrend_huber_spline(time, flux, mask, knot_distance):
    """Robust B-Spline regression with scikit-learn"""
    try:
        from sklearn.base import TransformerMixin
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import HuberRegressor
    except:
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
        return np.full(len(time), np.nan)
    else:
        pipeline = make_pipeline(BSplineFeatures(knots), HuberRegressor())
        trend = pipeline.fit(masked_time[:, np.newaxis], masked_flux)
        return trend.predict(time[:, None])
