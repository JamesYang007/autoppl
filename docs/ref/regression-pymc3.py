import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

basic_model = pm.Model()

xdata = np.array([2.4, 3.1, 3.6, 4, 4.5, 5.])
ydata = np.array([3.5, 4, 4.4, 5.01, 5.46, 6.1])

with basic_model:
    w = pm.Uniform('w', 0, 2)
    b = pm.Uniform('b', 0, 2)
    y = pm.Normal('y', mu=w * xdata + b, sigma=0.5, observed=ydata)

with basic_model:
    data = pm.sample(1000)

pm.traceplot(data)
plt.show()

print("w: ", data['w'].mean())
print("b: ", data['b'].mean())
