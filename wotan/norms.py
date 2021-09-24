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





def estimate_location(data, norm, maxiter=30, tol=1e-6):

    if norm == "huber":
        norm = HuberT()
    elif norm == "ramsay":
        norm = RamsayE()
    elif norm == "hampel":
        norm = Hampel()
    location = np.median(data)  # Initial guess
    #print(location)

    for iter in range(maxiter):
        W = norm.weights((data - location))
        #print(W)
        new_location = np.sum(W * data) / np.sum(W)
        #print(iter, new_location)
        if np.alltrue(np.less(np.fabs(location - new_location), tol)):
            return new_location
        else:
            location = new_location
    # Did not converge
    print(norm, 'Not converged')
    return np.median(data)



def ramsay(data, maxiter=30, tol=1e-6):
    
    a = 0.3
    location = np.median(data)  # Initial guess
    for iter in range(maxiter):
        z = data - location
        W = np.exp(-a * np.fabs(z))
        new_location = np.sum(W * data) / np.sum(W)
        #print(iter, new_location)
        if np.alltrue(np.less(np.fabs(location - new_location), tol)):
            return new_location
        else:
            location = new_location
    return np.median(data)



def huber(data, maxiter=30, tol=1e-6):
    
    t = 1.5
    location = np.median(data)  # Initial guess
    for iter in range(maxiter):
        z = data - location

        test = np.less_equal(np.fabs(z), t)
        absz = np.fabs(z)
        absz[test] = 1
        absz[absz==0] = 1e-100
        W = test + (1 - test) * t / absz

        new_location = np.sum(W * data) / np.sum(W)
        #print(iter, new_location)
        if np.alltrue(np.less(np.fabs(location - new_location), tol)):
            return new_location
        else:
            location = new_location
    return np.median(data)


def hampel(data, maxiter=30, tol=1e-6):
    
    a=1.7
    b=3.4
    c=8.5

    location = np.median(data)  # Initial guess
    for iter in range(maxiter):
        z = data - location

        fabs = np.fabs(z)
        fabs[fabs==0] = 1e-100
        W = (
            np.less_equal(z, a) + np.less_equal(z, b) * np.greater(z, a) * a / fabs
            + np.less_equal(z, c) * np.greater(z, b) * a * (c - fabs) / (fabs * (c - b))
            )
        W[np.where(np.isnan(W))] = 1
        

        new_location = np.sum(W * data) / np.sum(W)
        #print(iter, new_location)
        if np.alltrue(np.less(np.fabs(location - new_location), tol)):
            return new_location
        else:
            location = new_location
    return np.median(data)




data = [0, 1, 2, 3, 11, 11.5]

print('ramsay old')
print(estimate_location(data, norm="ramsay"))
print('ramsay new')
print(ramsay(data))

print('huber old')
print(estimate_location(data, norm="huber"))
print('huber new')
print(huber(data))

print('hampel old')
print(estimate_location(data, norm="hampel"))
print('hampel new')
print(hampel(data))
