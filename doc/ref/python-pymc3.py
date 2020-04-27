import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

basic_model = pm.Model()

with basic_model:
    mu = pm.Uniform('mu', -1, 1)
    sigma = pm.Uniform('sigma', 0, 1)
    x = pm.Normal('x', mu=mu, sigma=sigma, observed=[0.1, 0.2, 0.3, 0.4, 0.5])

with basic_model:
    data = pm.sample(1000)

pm.traceplot(data)
plt.show()

print("mu: ", data['mu'].mean())
print("sigma: ", data['sigma'].mean())
