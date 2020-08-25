from cmdstanpy import CmdStanModel
import numpy as np

y = np.array([1., -1.])
V = np.array([[2,1],[1,2]])
cool_dat = { 'N' : 2, 'y' : y.tolist(), 'V' : V.tolist() }

stan_file = 'nuts_wishart_cov_stan.stan'
sm = CmdStanModel(stan_file=stan_file)
fit = sm.sample(data=cool_dat, chains=1, cores=1,
                iter_warmup=50000, iter_sampling=50000, thin=1,
                max_treedepth=10, metric='diag', adapt_engaged=True,
                output_dir='.')
print(fit.summary())
