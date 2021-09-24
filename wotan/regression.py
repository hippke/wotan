from __future__ import division
import numpy as np


def regression(time, flux, method, window_length, cval):
    """Regressions with scikit-learn"""

    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.base import BaseEstimator, TransformerMixin
    except:
        raise ImportError('Could not import sklearn')

    class GF(BaseEstimator, TransformerMixin):
        """Uniformly spaced Gaussian features for one-dimensional input"""
    
        def __init__(self, N, width_factor=2.0):
            self.N = N
            self.width_factor = width_factor
        
        @staticmethod
        def _gauss_basis(x, y, width, axis=None):
            arg = (x - y) / width
            return np.exp(-0.5 * np.sum(arg ** 2, axis))
            
        def fit(self, X, y=None):
            # create N centers spread along the data range
            self.centers_ = np.linspace(X.min(), X.max(), self.N)
            self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
            return self
            
        def transform(self, X):
            return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                     self.width_, axis=1)

    offset = 1e-6  # lasso and elasticnet are less sensitive than ridge regression
    tol = 1e-6
    max_iter = 1e5
    if method == 'ridge':
        regression_method = Ridge(alpha=cval, max_iter=max_iter, tol=tol)
    elif method == 'lasso':
        regression_method = Lasso(alpha=cval * offset, max_iter=max_iter, tol=tol)
    elif method == 'elasticnet':
        regression_method = ElasticNet(alpha=cval * offset, max_iter=max_iter, tol=tol)

    duration = np.max(time) - np.min(time)
    no_knots = int(duration / window_length)
    X = time[:, np.newaxis]
    trend = make_pipeline(GF(no_knots), regression_method).fit(X, flux).predict(X)

    return trend
