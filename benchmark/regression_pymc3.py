import pymc3 as pm
import pandas as pd
import numpy as np

df = pd.read_csv("life-clean.csv", names=['le', 'alc', 'hiv', 'gdp'], delimiter=' ')
X, y = np.array(df[['alc', 'hiv', 'gdp']]), np.array(df['le'])

basic_model = pm.Model()

n_cols = np.size(X, 1)

with basic_model:
    a = pm.Normal('a', mu=0, sigma=5)
    b = pm.MvNormal('b', mu=np.zeros(n_cols),
                    cov=5*np.identity(n_cols),
                    shape=(1,n_cols))
    y_data = pm.Normal('y_data', mu=(pm.math.dot(X, b.T) + a),
                       sigma=5, observed=y)

with basic_model:
    data = pm.sample(draws=1000, n_init=1000,
                     chains=1, cores=1, tune=1000)

print(data)
