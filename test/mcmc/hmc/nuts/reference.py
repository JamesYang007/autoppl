import numpy as np
from scipy.stats import norm, uniform, bernoulli
from scipy.integrate import quad

x = np.array([2.5, 3, 3.5, 4, 4.5, 5.])
y = np.array([3.5, 4, 4.5, 5, 5.5, 6.])
q = np.array([2.4, 3.1, 3.6, 4, 4.5, 5.])
r = np.array([3.5, 4, 4.4, 5.01, 5.46, 6.1])

def nuts_sample_unif_normal_posterior_mean():
    x = 3.
    def p(w):
        prior = uniform.pdf(w, loc=-20, scale=40)
        likl = norm.pdf(x, w, 1.)
        return prior * likl
    norm_constant = quad(p, -20, 20)[0]
    mean = quad(lambda w: w*p(w), -20, 20)[0]
    return mean / norm_constant

def nuts_sample_regression_dist_weight():
    def p(w):
        prior = norm.pdf(w, loc=0, scale=2)
        likl = np.prod(norm.pdf(y, loc=x*w+1., scale=0.5))
        return prior * likl
    norm_constant = quad(p, -np.inf, np.inf)[0]
    mean = quad(lambda w: w*p(w), -np.inf, np.inf)[0]
    return mean/norm_constant

def nuts_sample_regression_dist_weight_bias():
    def p(w,b):
        prior_b = norm.pdf(b, loc=0, scale=2)
        prior_w = norm.pdf(w, loc=0, scale=2)
        likl = np.prod(norm.pdf(y, loc=x*w+b, scale=0.5))
        return prior_b * prior_w * likl
    norm_constant = quad(lambda b: quad(lambda w: p(w,b), -np.inf, np.inf)[0],
                         -np.inf, np.inf)[0]
    mean_w = quad(lambda b: quad(lambda w: w*p(w,b), -np.inf, np.inf)[0],
                  -np.inf, np.inf)[0]
    mean_b = quad(lambda b: quad(lambda w: b*p(w,b), -np.inf, np.inf)[0],
                  -np.inf, np.inf)[0]
    return (mean_b/norm_constant, mean_w/norm_constant)

def nuts_sample_regression_dist_uniform():
    def p(w,b):
        prior_w = uniform.pdf(w, loc=0, scale=2)
        prior_b = uniform.pdf(b, loc=0, scale=2)
        likl = np.prod(norm.pdf(y, loc=x*w+b, scale=0.5))
        return prior_w * prior_b * likl
    norm_constant = quad(lambda b: quad(lambda w: p(w,b), 0,2)[0],
                         0,2)[0]
    mean_w = quad(lambda b: quad(lambda w: w*p(w,b), 0,2)[0],
                  0,2)[0]
    mean_b = quad(lambda b: quad(lambda w: b*p(w,b), 0,2)[0],
                  0,2)[0]
    return (mean_w/norm_constant, mean_b/norm_constant)

def nuts_sample_regression_fuzzy_uniform():
    def p(w,b):
        prior_w = uniform.pdf(w, loc=0, scale=2)
        prior_b = uniform.pdf(b, loc=0, scale=2)
        likl = np.prod(norm.pdf(r, loc=q*w+b, scale=0.5))
        return prior_w * prior_b * likl
    norm_constant = quad(lambda b: quad(lambda w: p(w,b), 0,2)[0],
                         0,2)[0]
    mean_w = quad(lambda b: quad(lambda w: w*p(w,b), 0,2)[0],
                  0,2)[0]
    mean_b = quad(lambda b: quad(lambda w: b*p(w,b), 0,2)[0],
                  0,2)[0]
    return (mean_w/norm_constant, mean_b/norm_constant)

def nuts_sample_regression_dot():
    x = np.array([1,-1,0.5])
    y = np.array([2,-0.13,1.32])
    def p(w,b):
        prior_w = uniform.pdf(w, loc=0, scale=2)
        prior_b = uniform.pdf(b, loc=0, scale=2)
        likl = np.prod(norm.pdf(y, loc=x*w+b, scale=0.5))
        return prior_w * prior_b * likl
    norm_constant = quad(lambda b: quad(lambda w: p(w,b), 0,2)[0],
                         0,2)[0]
    mean_w = quad(lambda b: quad(lambda w: w*p(w,b), 0,2)[0],
                  0,2)[0]
    mean_b = quad(lambda b: quad(lambda w: b*p(w,b), 0,2)[0],
                  0,2)[0]
    return (mean_w/norm_constant, mean_b/norm_constant)

def nuts_coin_flip():
    x = np.array([0,1,1])
    def p(t):
        prior_t = uniform.pdf(t, loc=0, scale=1)
        likl = np.prod(bernoulli.pmf(x, p=t))
        return prior_t * likl
    norm_constant = quad(p, 0, 1)[0]
    mean = quad(lambda t: t*p(t), 0, 1)[0]
    return mean/norm_constant

def nuts_mean_vec_stddev_vec():
    x = np.array([2.5, 3])
    y = np.array([3.5, 4])
    def p(s1, s2, w, b):
        prior_s1 = uniform.pdf(s2, loc=0.5, scale=4.5)
        prior_s2 = uniform.pdf(s1, loc=0.5, scale=4.5)
        prior_w = uniform.pdf(w, loc=0., scale=2.)
        prior_b = uniform.pdf(b, loc=0., scale=2.)
        likl = np.prod(norm.pdf(y, loc=x*w+b, scale=[s1,s2]))
        return prior_s1 * prior_s2 * prior_w * prior_b * likl

    integrator = lambda f: \
        quad(lambda s1: \
            quad(lambda s2: \
                 quad(lambda w: \
                      quad(lambda b: f(s1,s2,w,b), 0, 2)[0],
                      0, 2)[0],
                 0.5, 5)[0],
            0.5, 5.)[0]

    norm_constant = integrator(p)
    mean_s1 = integrator(lambda s1,s2,w,b: s1*p(s1,s2,w,b))
    mean_s2 = integrator(lambda s1,s2,w,b: s2*p(s1,s2,w,b))
    mean_w = integrator(lambda s1,s2,w,b: w*p(s1,s2,w,b))
    mean_b = integrator(lambda s1,s2,w,b: b*p(s1,s2,w,b))

    return np.array([mean_s1, mean_s2, mean_w, mean_b]) / norm_constant

if __name__ == '__main__':
    res = nuts_mean_vec_stddev_vec()
    print(res)
