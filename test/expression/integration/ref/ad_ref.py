import numpy as np
from scipy.stats import norm

def logit(u):
    return np.log(u/(1-u))

def inv_logit(v):
    return 1./(1. + np.exp(-v))

def dinv_logit(v):
    return inv_logit(v) * (1 - inv_logit(v))

# derivative of constrained phi w.r.t. unconstrained phi
def dphi(Phi, r):
    return r * dinv_logit(Phi)

def dsigma(Sigma):
    return np.exp(Sigma)

def stochastic_volatility():
    y = np.array([0.4, 0.5])
    phi = 0.95
    sigma = 0.25
    mu = -1.02
    h_std = np.array([1,-1])

    def compute_h(phi, sigma, mu, h_std):
        h = np.zeros(len(h_std))
        h[0] = h_std[0] * sigma / np.sqrt(1 - phi**2) + mu
        h[1] = h_std[1] * sigma + mu + phi * (h[0] - mu)
        return h

    def eval(y, phi, sigma, mu, h_std):
        std_correction = len(h_std) / 2 * np.log(2 * np.pi)
        y_correction = len(y) / 2 * np.log(2 * np.pi)
        h = compute_h(phi, sigma, mu, h_std)
        lpdf = -np.log(2)                                   \
               + np.log((phi + 1) * (1 - (phi + 1)/2)) \
               -np.log(5 + sigma**2 / 5)                    \
               + np.log(sigma)  \
               -np.log(10 + mu**2 / 10)                     \
               + np.sum(norm.logpdf(h_std, 0, 1))           \
               + std_correction                             \
               + np.sum(norm.logpdf(y, 0, np.exp(h/2)))     \
               + y_correction

        return [lpdf]

    def dPhi(y, phi, sigma, mu, h_std):
        Phi = logit((phi + 1) / 2)
        h = compute_h(phi, sigma, mu, h_std)
        dh = np.zeros(len(h_std))
        dh[0] = h_std[0] * sigma * phi / (1 - phi**2)**(1.5) * dphi(Phi, 2)
        dh[1] = dphi(Phi, 2) * (h[0] - mu) + phi * dh[0]
        out = 1 - 2 * inv_logit(Phi) - 0.5 * np.sum((1 - y**2 * np.exp(-h)) * dh)
        return [out]

    def dSigma(y, phi, sigma, mu, h_std):
        Sigma = np.log(sigma)
        h = compute_h(phi, sigma, mu, h_std)
        dh = np.zeros(len(h_std))
        dh[0] = h_std[0] / (1 - phi**2)**(0.5) * dsigma(Sigma)
        dh[1] = h_std[1] * dsigma(Sigma) + phi * dh[0]
        out = -2 * sigma**2 / (5**2 + sigma**2) + 1 - 0.5 * np.sum((1 - y**2 * np.exp(-h)) * dh)
        return [out]

    def dMu(y, phi, sigma, mu, h_std):
        h = compute_h(phi, sigma, mu, h_std)
        dh = np.zeros(len(h_std))
        dh[0] = 1
        dh[1] = 1 + phi * (dh[0] - 1)
        out = -2 * mu / (10**2 + mu**2) - 0.5 * np.sum((1 - y**2 * np.exp(-h)) * dh)
        return [out]

    def dh_std(y, phi, sigma, mu, h_std):
        h = compute_h(phi, sigma, mu, h_std)
        dh = np.zeros(len(h_std))
        dh_std = np.zeros(len(h_std))
        dh[0] = sigma / np.sqrt(1 - phi**2)
        dh[1] = phi * dh[0]
        dh_std[0] = -h_std[0] - 0.5 * np.sum((1 - y**2 * np.exp(-h)) * dh)

        dh[0] = 0
        dh[1] = sigma
        dh_std[1] = -h_std[1] - 0.5 * np.sum((1 - y**2 * np.exp(-h)) * dh)
        return dh_std

    return dh_std(y, phi, sigma, mu, h_std)

if __name__ == '__main__':
    res = stochastic_volatility()

    for r in res:
        print("{0:.18f}".format(r))
