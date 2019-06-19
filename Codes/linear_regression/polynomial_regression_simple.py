import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import machine_learning.linear_regression.lib.linear_regression as lib

def generate_samples(m):
        X = 2 * np.random.rand(m, 1) 
        y = X**2 - 2 * X + 1 + np.random.normal(0, 0.1, (m,1))
        return X, y

np.random.seed(0)
X, y = generate_samples(100)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = lib.LinearRegression()
model.fit(X_poly, y)

plt.figure(0)
plt.scatter(X, y)
plt.figure(1)
plt.scatter(X, y)
W = np.linspace(0, 2, 100).reshape(100, 1)
W_poly = poly.fit_transform(W)
u = model.predict(W_poly)
plt.plot(W, u)
plt.show()




