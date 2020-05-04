import pystan
import pandas as pd
import numpy as np

cool_code = """
data {
  int N;
  vector[N] le;
  vector[N] alc;
  vector[N] hiv;
  vector[N] gdp;
}
parameters {
  real ple;
  real palc;
  real phiv;
  real pgdp;
}
model {
  ple ~ normal(0,5);
  palc ~ normal(0,5);
  phiv ~ normal(0,5);
  pgdp ~ normal(0,5);
  le ~ normal(ple + alc * palc + hiv * phiv + gdp * pgdp, 5);
}
"""

df = pd.read_csv("life-clean.csv", names=['le', 'alc', 'hiv', 'gdp'], delimiter=' ')
N = df.shape[0]
cool_dat = {
    'N' : N,
    'le' : df['le'],
    'alc' : df['alc'],
    'hiv' : df['hiv'],
    'gdp' : df['gdp']
}

sm = pystan.StanModel(model_code=cool_code)
fit = sm.sampling(data=cool_dat, chains=1, n_jobs=1,
                  # (default) iter=2000, warmup=iter//2
                  init='random',
                  control={
                      'adapt_delta' : 0.95,
                      'stepsize_jitter' : 0,
                      'metric' : 'unit_e',
                      'max_treedepth' : 10
                  },
                  init_r=5)
print(fit)
