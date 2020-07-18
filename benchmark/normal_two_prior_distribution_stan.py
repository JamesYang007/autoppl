from cmdstanpy import CmdStanModel
import numpy as np

N = 1000
y = np.random.normal(size = N)
cool_dat = { 'N' : N, 'y' : list(y) }

stan_file = 'normal_two_prior_distribution_stan.stan'
sm = CmdStanModel(stan_file=stan_file)
fit = sm.sample(data=cool_dat, chains=1, cores=1,
                iter_warmup=7000, iter_sampling=7000, thin=1,
                max_treedepth=10, metric='diag', adapt_engaged=True,
                output_dir='.')
print(fit.summary())
