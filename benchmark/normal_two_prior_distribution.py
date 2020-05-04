import pystan
import numpy as np

cool_code = """
data {
  int N;
  real y[N];
}
parameters {
  real lambda1;
  real lambda2;
  real<lower=0> sigma;
}
transformed parameters {
  real mu;
  mu = lambda1 + lambda2;
}
model {
  sigma ~ uniform(0, 20);
  lambda1 ~ normal(0, 10);
  lambda2 ~ normal(0, 10);
  y ~ normal(mu, sigma);
}
"""

N = 100
y = np.random.normal(size = N)
cool_dat = { 'N' : N, 'y' : y }

sm = pystan.StanModel(model_code=cool_code)
fit = sm.sampling(data=cool_dat, chains=1, n_jobs=1)
print(fit)
