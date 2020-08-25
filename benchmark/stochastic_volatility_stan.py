from cmdstanpy import CmdStanModel
import numpy as np

n_data = 500

actual_phi = 0.95;
actual_sigma = 0.25;
actual_mu = -1.02;

y = np.zeros(n_data)

actual_h = np.random.normal() * actual_sigma / \
            np.sqrt(1. - actual_phi * actual_phi) \
            + actual_mu
y[0] = np.random.normal() * np.exp(actual_h/2.)

for i in range(1, n_data):
    actual_h = np.random.normal() * actual_sigma + actual_mu + actual_phi * (actual_h - actual_mu)
    y[i] = np.random.normal() * np.exp(actual_h/2.)

data = {
    'T' : len(y),
    'y' : y,
}

stan_file = "stochastic_volatility_stan.stan"
sm = CmdStanModel(stan_file=stan_file)

sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]

fit = sm.sample(data=data, chains=1, cores=1,
                iter_warmup=sizes[3], iter_sampling=sizes[3], thin=1,
                max_treedepth=10, metric='diag', adapt_engaged=True,
                output_dir='.')
print(fit.summary())
