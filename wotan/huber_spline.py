from __future__ import print_function, division
import numpy
import scipy.interpolate


def detrend_huber_spline(time, flux, knot_distance):
    """Robust B-Spline regression with scikit-learn"""
    try:
        from sklearn.base import TransformerMixin
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import HuberRegressor
    except:
        raise ImportError('Could not import sklearn')

    class BSplineFeatures(TransformerMixin):
        def __init__(self, knots, degree=3, periodic=False):
            self.bsplines = self.get_bspline_basis(
                knots, degree, periodic=periodic
            )
            self.nsplines = len(self.bsplines)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            nsamples, nfeatures = X.shape
            features = numpy.zeros(
                (nsamples, nfeatures * self.nsplines)
            )
            for ispline, spline in enumerate(self.bsplines):
                istart = ispline * nfeatures
                iend = (ispline + 1) * nfeatures
                features[
                    :, istart:iend
                ] = scipy.interpolate.splev(X, spline)
            return features

        def get_bspline_basis(
            self, knots, degree=3, periodic=False
        ):
            """Get spline coefficients for each basis spline"""
            knots, coeffs, degree = scipy.interpolate.splrep(
                knots,
                numpy.zeros(len(knots)),
                k=degree,
                per=periodic,
            )
            ncoeffs = len(coeffs)
            bsplines = []
            for ispline in range(len(knots)):
                coeffs = [
                    1.0 if ispl == ispline else 0.0
                    for ispl in range(ncoeffs)
                ]
                bsplines.append((knots, coeffs, degree))
            return bsplines

    duration = numpy.max(time) - numpy.min(time)
    no_knots = int(duration / knot_distance)
    knots = numpy.linspace(
        numpy.min(time), numpy.max(time), no_knots
    )

    trend = (
        make_pipeline(
            BSplineFeatures(knots), HuberRegressor()
        )
        .fit(time[:, numpy.newaxis], flux)
        .predict(time[:, None])
    )
    return trend
