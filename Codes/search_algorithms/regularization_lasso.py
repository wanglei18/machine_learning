import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from machine_learning.search_algorithms.lib.lasso import Lasso
import matplotlib.pyplot as plt

def generate_samples(m):
    X = 2 * (np.random.rand(m, 1) - 0.5) 
    y = X + np.random.normal(0, 0.3, (m,1))
    return X, y

np.random.seed(100)
X, y = generate_samples(10)
poly = PolynomialFeatures(degree = 10)
X_poly = poly.fit_transform(X)
model = Lasso(Lambda=0.001)
model.fit(X_poly, y, eta=0.01, N=50000)

plt.axis([-1, 1, -2, 2])
plt.scatter(X, y)
W = np.linspace(-1, 1, 100).reshape(100, 1)
W_poly = poly.fit_transform(W)
u = model.predict(W_poly)
plt.plot(W, u)
plt.show()

   







