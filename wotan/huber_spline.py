from __future__ import print_function, division
import warnings
import numpy as np
import scipy.interpolate
import scipy.sparse


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
            knots, _, degree = self.bsplines[0]
            try:
                # All basis functions at once (same values as one splev per basis
                # function; the last degree + 1 columns stay zero, as with splev)
                design = scipy.interpolate.BSpline.design_matrix(
                    np.ravel(X), knots, degree, extrapolate=True)
                features[:, :design.shape[1]] = design.toarray()
            except (AttributeError, TypeError):  # scipy < 1.10
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
        y = (masked_flux - offset) / scale
        try:
            return _sparse_huber_spline(time, masked_time, y, knots, cval) * scale + offset
        except (AttributeError, TypeError):  # scipy < 1.10: dense pipeline below
            pass
        # StandardScaler on the spline features and max_iter=1000 improve convergence;
        # ``cval`` is the epsilon of the HuberRegressor (default 1.35)
        pipeline = make_pipeline(
            BSplineFeatures(knots), StandardScaler(), HuberRegressor(max_iter=1000, epsilon=cval)
        )
        trend = pipeline.fit(masked_time[:, np.newaxis], y)
        return trend.predict(time[:, None]) * scale + offset


def _huber_loss_and_gradient(w, X, Xt, y, epsilon, alpha):
    """Loss and gradient of scikit-learn's HuberRegressor (unit sample weights, with
    intercept), for a sparse X: the gradient is X^T g for a per-point vector g, so no
    rows of X have to be selected and copied"""
    n_features = X.shape[1]
    intercept = w[-2]
    sigma = w[-1]
    coef = w[:n_features]
    linear_loss = y - X @ coef - intercept
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > epsilon * sigma
    num_outliers = np.count_nonzero(outliers_mask)
    outlier_loss = (2.0 * epsilon * np.sum(abs_linear_loss[outliers_mask])
                    - sigma * num_outliers * epsilon ** 2)
    non_outliers = np.where(outliers_mask, 0.0, linear_loss)
    squared_loss = np.dot(non_outliers, non_outliers) / sigma
    signed_outliers = np.where(outliers_mask, np.sign(linear_loss), 0.0)
    grad = np.zeros(n_features + 2)
    grad[:n_features] = Xt @ (-2.0 / sigma * non_outliers - 2.0 * epsilon * signed_outliers)
    grad[:n_features] += alpha * 2.0 * coef
    grad[-1] = len(y) - num_outliers * epsilon ** 2 - squared_loss / sigma
    grad[-2] = -2.0 * np.sum(non_outliers) / sigma - 2.0 * epsilon * np.sum(signed_outliers)
    loss = len(y) * sigma + squared_loss + outlier_loss + alpha * np.dot(coef, coef)
    return loss, grad


def _huber_fit(X, y, epsilon, alpha=0.0001, tol=1e-05, max_iter=1000):
    """scikit-learn's HuberRegressor.fit (same start, bounds and L-BFGS-B options)
    with the sparse loss above. Returns coefficients and intercept."""
    from scipy import optimize
    parameters = np.zeros(X.shape[1] + 2)
    parameters[-1] = 1
    bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10
    opt_res = optimize.minimize(
        _huber_loss_and_gradient, parameters, method="L-BFGS-B", jac=True,
        args=(X, X.T.tocsr(), y, epsilon, alpha),
        options={"maxiter": max_iter, "gtol": tol}, bounds=bounds)
    if opt_res.status == 2:
        raise ValueError("hspline: Huber regression failed: %s" % opt_res.message)
    if opt_res.status == 1:
        warnings.warn("hspline: Huber regression did not converge in %d iterations"
                      % max_iter, UserWarning)
    return opt_res.x[:X.shape[1]], opt_res.x[-2]


def _sparse_huber_spline(time, masked_time, y, knots, cval):
    """The Huber B-spline regression of ``detrend_huber_spline`` with a sparse design
    matrix: each point lies on only 4 B-splines, so the matrix is banded. The
    features are scaled to unit standard deviation like StandardScaler, but not
    centred (centring would make the matrix dense). The model is the same: the
    centring only moves the unpenalised intercept, so the optimum is the same and the
    fit agrees with the dense pipeline to the optimiser tolerance (~1e-7 relative)."""
    spline_knots, _, degree = scipy.interpolate.splrep(knots, np.zeros(len(knots)))

    def design(x):
        return scipy.interpolate.BSpline.design_matrix(
            x, spline_knots, degree, extrapolate=True).tocsr()

    features = design(masked_time)
    mean = np.asarray(features.mean(axis=0)).ravel()
    mean_sq = np.asarray(features.multiply(features).mean(axis=0)).ravel()
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    std[std == 0] = 1  # as StandardScaler: constant features are not scaled
    inv_std = scipy.sparse.diags(1 / std)
    coef, intercept = _huber_fit((features @ inv_std).tocsr(), y, cval)
    return design(time) @ (inv_std @ coef) + intercept
