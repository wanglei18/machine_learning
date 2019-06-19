import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from machine_learning.linear_regression.lib.ridge_regression import RidgeRegression

def generate_samples(m):
    X = 2 * (np.random.rand(m, 1) - 0.5) 
    y = X + np.random.normal(0, 0.3, (m,1))
    return X, y

np.random.seed(100)
poly = PolynomialFeatures(degree = 10)

X, y = generate_samples(10)
X_poly = poly.fit_transform(X)
model = RidgeRegression(Lambda = 0.01)
model.fit(X_poly, y)

plt.scatter(X, y)
plt.axis([-1, 1, -2, 2])
W = np.linspace(-1, 1, 100).reshape(100, 1)
W_poly = poly.fit_transform(W)
u = model.predict(W_poly)
plt.plot(W, u)
plt.show()
   







