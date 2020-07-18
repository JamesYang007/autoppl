import numpy as np
from scipy.stats import norm

x_val = -0.2
x_vec = np.array([0., 1., 2.])
mean_val = 0.
mean_vec = np.array([-1., 0., 1.])
sd_val = 1.
sd_vec = np.array([1., 2., 3.])

# xyz naming has the following convention:
# x,y,z are either s (scalar) or v (vector)
# x: whether x is scalar or vector
# y: whether mean is scalar or vector
# z: whether sd is scalar or vector

def correction(n):
    return n/2.*np.log(2.*np.pi)

def sss():
    log_pdf = np.sum(norm.logpdf(x_val, loc=mean_val, scale=sd_val)) \
            + correction(1)
    return log_pdf

def vss():
    log_pdf = np.sum(norm.logpdf(x_vec, loc=mean_val, scale=sd_val)) \
            + correction(len(x_vec))
    return log_pdf

def vsv():
    log_pdf = np.sum(norm.logpdf(x_vec, loc=mean_val, scale=sd_vec)) \
            + correction(len(sd_vec))
    return log_pdf

def vvs():
    log_pdf = np.sum(norm.logpdf(x_vec, loc=mean_vec, scale=sd_val)) \
            + correction(len(x_vec))
    return log_pdf

def vvv():
    log_pdf = np.sum(norm.logpdf(x_vec, loc=mean_vec, scale=sd_vec)) \
            + correction(len(x_vec))
    return log_pdf

if __name__ == '__main__':
    print('sss', sss())
    print('vss', vss())
    print('vsv', vsv())
    print('vvs', vvs())
    print('vvv', vvv())
