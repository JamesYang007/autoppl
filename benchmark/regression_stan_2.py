from cmdstanpy import CmdStanModel
import pandas as pd
import numpy as np

N = 30000
X = np.random.normal(loc=[-1, 0, 1], scale=1.4, size=(N, 3))
w_true = np.array([1.4, 2., 0.32])
y = X.dot(w_true) + np.random.normal(loc=0., scale=1.0, size=N)

cool_dat = {
    'N' : N,
    'x1' : list(X[:,0]),
    'x2' : list(X[:,1]),
    'x3' : list(X[:,2]),
    'y' : list(y)
}

stan_file = "regression_stan_2.stan"
sm = CmdStanModel(stan_file=stan_file)
fit = sm.sample(data=cool_dat, chains=1, cores=1,
                iter_warmup=1000, iter_sampling=1000, thin=1,
                max_treedepth=10, metric='diag', adapt_engaged=True,
                output_dir='.')
print(fit.summary())
