from __future__ import division
import numpy as np


class RobustNorm(object):
    def __call__(self, z):
        return self.rho(z)


class HuberT(RobustNorm):
    def __init__(self, t=1.5):
        self.t = t

    def _subset(self, z):
        return np.less_equal(np.fabs(z), self.t)

    def weights(self, z):
        test = self._subset(z)
        absz = np.fabs(z)
        absz[test] = 1.0
        #absz[absz==0] = 1e-10
        return test + (1 - test) * self.t / absz


class RamsayE(RobustNorm):
    def __init__(self, a=0.3):
        self.a = a

    def weights(self, z):
        return np.exp(-self.a * np.fabs(z))


class Hampel(RobustNorm):
    def __init__(self, a=1.7, b=3.4, c=8.5):
        self.a = a
        self.b = b
        self.c = c

    def _subset(self, z):
        z = np.fabs(z)
        t1 = np.less_equal(z, self.a)
        t2 = np.less_equal(z, self.b) * np.greater(z, self.a)
        t3 = np.less_equal(z, self.c) * np.greater(z, self.b)
        return t1, t2, t3

    def weights(self, z):
        a = self.a
        b = self.b
        c = self.c
        t1, t2, t3 = self._subset(z)
        fabs = np.fabs(z)
        fabs[fabs==0] = 1e-100
        v = (
            t1
            + t2 * a / fabs
            + t3
            * a
            * (c - fabs)
            / (fabs * (c - b))
        )
        v[np.where(np.isnan(v))] = 1
        return v


def estimate_location(a, norm, maxiter=30, tol=1e-06):

    if norm == "huber":
        norm = HuberT()
    elif norm == "ramsay":
        norm = RamsayE()
    elif norm == "hampel":
        norm = Hampel()
    location = np.median(a)  # Initial guess
    print(location)

    for iter in range(maxiter):
        W = norm.weights((a - location))
        #print(W)
        new_location = np.sum(W * a) / np.sum(W)
        print(iter, new_location)
        if np.alltrue(np.less(np.fabs(location - new_location), tol)):
            return new_location
        else:
            location = new_location
    # Did not converge
    print(norm, 'Not converged')
    return np.median(a)
