import pymc3 as pm
import numpy as np

y_data = np.random.normal(size = 100)

basic_model = pm.Model()

with basic_model:
    sigma = pm.Uniform('sigma', 0, 20)
    l1 = pm.Normal('l1', 0, 10)
    l2 = pm.Normal('l2', 0, 10)
    y = pm.Normal('y', mu=(l1 + l2), sigma=sigma, observed=y_data)

with basic_model:
    data = pm.sample(draws=1000, n_init=1000, chains=1, cores=1, tune=1000)

print(data)
