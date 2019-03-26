import numpy
from numpy import exp, pi, sin
import scipy.interpolate
#import george
#from george import kernels
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor
from statsmodels.robust.scale import huber, Huber
#from statsmodels.robust import norms, scale
#import statsmodels.api as sm
from statsmodels.robust.scale import huber
from astropy.stats import median_absolute_deviation
import numba
from supersmoother import SuperSmoother


@numba.jit(fastmath=True, parallel=False, nopython=True)
def biweight_location_iter(data, c=6.0, ftol=1e-6, maxiter=15):
    """Robust location estimate using iterative Tukey's biweight"""

    delta_center = 1e100
    niter = 0

    # Initial estimate for the central location
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    center = center_old = numpy.median(data)
    if mad == 0:
        return center

    while not niter >= maxiter and not (abs(delta_center) <= ftol):
        # New estimate for center
        distance = data - center
        #print(distance, c, mad)
        weight = distance / (c * mad)
        mask = (numpy.abs(weight) >= 1)
        weight = (1 - weight ** 2) ** 2
        weight[mask] = 0
        center += numpy.sum(distance * weight) / numpy.sum(weight)

        # Calculate how much center moved to check convergence threshold
        delta_center = center_old - center
        center_old = center
        niter += 1
    return center


def hodges(data):
    # Hodges–Lehmann–Sen estimator
    i = 0
    j = 0
    hodges = []
    while i < len(data):
        while j < len(data):
            hodges.append(numpy.mean([data[i], data[j]]))
            j = j + 1
        i = i + 1
        j = i
    return numpy.median(hodges)


def welsh(data, c=2.11):
    # Welsch loss aka Leclerc
    M = numpy.median(data)
    d = data - M
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    if mad == 0:
        return M
    u = d / (c * mad)
    mask = (numpy.abs(u) >= 1)
    u = exp(-u**2 / 2)
    u[mask] = 0
    return M + (d * u).sum() / u.sum()


def andrewsinewave_location(data, c=1.339):  # 1.339
    M = numpy.median(data)
    d = data - M
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    if mad == 0:
        return M
    u = d / (c * mad)
    u[u==0] = 1e-100  # avoid division by zero
    mask = (numpy.abs(u) >= pi)
    #u = sin(u) / u
    u = sin(u) / u# / u
    u[mask] = 0
    return M + (d).sum() / u.sum()



def biweight_location(data, c=6.0):
    M = numpy.median(data)
    d = data - M
    mad = numpy.median(numpy.abs(data - numpy.median(data)))
    if mad == 0:
        return M
    u = d / (c * mad)
    mask = (numpy.abs(u) >= 1)
    u = (1 - u ** 2) ** 2
    u[mask] = 0
    return M + (d * u).sum() / u.sum()

"""
data = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
print(biweight_location(data, c=6))
print(biweight_location(data, c=4.685))
print(andrewsinewave_location(data))
print(welsh(data))
"""


class BSplineFeatures(TransformerMixin):
    """Robust B-Spline regression with scikit-learn"""

    def __init__(self, knots, degree=3, periodic=False):
        self.bsplines = self.get_bspline_basis(knots, degree, periodic=periodic)
        self.nsplines = len(self.bsplines)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = numpy.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.bsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def get_bspline_basis(self, knots, degree=3, periodic=False):
        """Get spline coefficients for each basis spline"""
        knots, coeffs, degree = scipy.interpolate.splrep(
            knots, numpy.zeros(len(knots)), k=degree, per=periodic)
        ncoeffs = len(coeffs)
        bsplines = []
        for ispline in range(len(knots)):
            coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
            bsplines.append((knots, coeffs, degree))
        return bsplines


#@numba.jit(fastmath=True, parallel=False, nopython=True)
def running_segment(t, y, window, method, c=6, kernel='ExpSquaredKernel'):
    mean_all = numpy.full(len(t), numpy.nan)
    for i in range(len(t)-1):
        if t[i] > min(t) + window / 2 and t[i] < max(t)-(window / 2):
            mean_segment = numpy.nan
            idx_start = numpy.argmax(t > t[i] - window/2)# - 1
            idx_end = numpy.argmax(t > t[i] + window/2)# - 1
            data_segment = y[idx_start:idx_end]
            if method=='andrew':
                mean_segment = andrewsinewave_location(data_segment)
            if method=='welsh':
                mean_segment = welsh(data_segment)
            if method=='hampel':
                mean_segment = hampel(data_segment, 2, [0.1])
            if method=='median':
                mean_segment = numpy.median(data_segment)
            if method=='mean':
                mean_segment = numpy.mean(data_segment)
            elif method=='trim_mean':
                mean_segment = scipy.stats.trim_mean(data_segment, proportiontocut=0.1)
            elif method=='biweight':
                mean_segment = biweight_location(data_segment, c=c)
            elif method=='biweight_iter':
                mean_segment = biweight_location_iter(data_segment, c=c)
            elif method=='hodges':
                mean_segment = hodges(data_segment)

            elif method=='huber1':  # fast but fails sometimes
                try:
                    mean_segment = huber(data_segment)[0]
                except:
                    mean_segment = numpy.nan
            elif method=='huber2':  # slow but ok
                #print('Detrending with huber2...', i)
                mean_segment = sm.RLM(
                    data_segment,
                    numpy.ones(len(data_segment)),
                    M=norms.HuberT(t=1.35)).fit(
                    scale_est=scale.HuberScale(d=1.35)).params
            #print(idx_start, t[idx_start], idx_end, t[idx_end], mean_segment, no_values)
            mean_all[i+int(window/2)] = mean_segment
    mean_all = numpy.roll(mean_all, int(window/2))
    return mean_all


def spline_segment(t, y, knot_distance):
    duration = max(t)-min(t)
    no_knots = int(duration / knot_distance)
    knots = numpy.linspace(numpy.min(t), numpy.max(t), no_knots)
    try:
        model = make_pipeline(
            BSplineFeatures(knots), HuberRegressor()).fit(t[:, numpy.newaxis], y)
        trend = model.predict(t[:, None])
    except:
        trend = numpy.full(len(t), numpy.nan)

    return trend


def gp(t, y, metric=10, oversample=1, kernel='ExpSquaredKernel'):

    y = y / numpy.median(y) - 1
    yerr = numpy.full(len(t) * oversample, 200 * 10**-6)#)
    if kernel=='ExpSquaredKernel':
        kernel = kernels.ExpSquaredKernel(metric)
    elif kernel=='Matern32Kernel':
        kernel = kernels.Matern32Kernel(metric)
    gp = george.GP(kernel)
    gp.compute(t, yerr)
    x_pred = numpy.linspace(min(t), max(t), oversample * len(t))
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    sampled_trend = []
    for i in range(len(t)):
        sampled_trend.append(pred[numpy.argmax(x_pred>t[i])-1])
    trend = numpy.array(sampled_trend) + 1
    return trend


def trend_supersmoother(t, y, window):
    win = window / (max(t) - min(t))
    model = SuperSmoother(
        primary_spans=(win/3, win, win*3),
        middle_span=3*win,
        final_span=win
        )
    model.fit(t, y)
    return model.predict(t)


def trend_lowess(t, y, window, max_iter=5):
    trend_lo = sm.nonparametric.lowess(
        y,
        t,
        return_sorted=False,
        it=max_iter,
        frac=window / (max(t) - min(t)))
    return trend_lo


#@numba.jit(fastmath=True, parallel=False, nopython=True)
def trend(t, y, window, method='mean', oversample=1, c=6, kernel='ExpSquaredKernel'):

    # refactor like line 227 in:
    # https://github.com/christinahedges/lightkurve/blob/master/lightkurve/lightcurve.py
    gaps = numpy.diff(t)
    gaps_indexes = numpy.where(gaps > window)
    gaps_indexes = numpy.add(gaps_indexes, 1)  # Off by one :-)
    gaps_indexes = numpy.concatenate(gaps_indexes).ravel()
    gaps_indexes = numpy.append(numpy.array([0]), gaps_indexes)  # Start
    gaps_indexes = numpy.append(gaps_indexes, numpy.array([len(t)+1]))  # End point
    #print(gaps_indexes)
    y_new = numpy.array([])
    trend = numpy.array([])
    trend_segment = numpy.array([])
    for i in range(len(gaps_indexes)-1):
        start_segment = gaps_indexes[i]
        end_segment = gaps_indexes[i+1]
        if method == 'huberspline':
            trend_segment = spline_segment(
                t[start_segment:end_segment], y[start_segment:end_segment],
                knot_distance=window)

        else:
            trend_segment = running_segment(
                t[start_segment:end_segment], y[start_segment:end_segment],
                window,
                c=c,
                method=method)

        trend = numpy.append(trend, trend_segment)
    return trend
