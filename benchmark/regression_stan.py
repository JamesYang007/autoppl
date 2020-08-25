from cmdstanpy import CmdStanModel
import pandas as pd
import numpy as np

df = pd.read_csv("life-clean.csv", names=['le', 'alc', 'hiv', 'gdp'], delimiter=' ')
X, y = np.array(df[['alc', 'hiv', 'gdp']]), np.array(df['le'])
cool_dat = {
    'N' : df.shape[0],
    'K' : df.shape[1] - 1,
    'x' : X,
    'y' : y,
}

stan_file = "regression_stan.stan"
sm = CmdStanModel(stan_file=stan_file)

sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]

fit = sm.sample(data=cool_dat, chains=1, cores=1,
                iter_warmup=sizes[6], iter_sampling=sizes[6], thin=1,
                max_treedepth=10, metric='diag', adapt_engaged=True,
                output_dir='.')
print(fit.summary())
