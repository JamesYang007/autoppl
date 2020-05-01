import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

basic_model = pm.Model()

xdata = np.arange(500)
ydata = 2 * xdata + np.random.randn(500) / 4 + 3

with basic_model:
    w = pm.Normal('w', mu=0, sigma=5)
    b = pm.Normal('b', mu=0, sigma=5)
    y = pm.Normal('y', mu=w * xdata + b, sigma=1.0, observed=ydata)

with basic_model:
    data = pm.sample(500)

pm.traceplot(data)
plt.show()

print("w: ", data['w'].mean())
print("b: ", data['b'].mean())
