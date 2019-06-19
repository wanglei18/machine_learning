import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from machine_learning.linear_regression.lib.stagewise_regression import StagewiseRegression

def generate_samples(m):
    X = 2 * (np.random.rand(m, 1) - 0.5) 
    y = X + np.random.normal(0, 0.3, (m,1))
    return X, y

np.random.seed(100)
poly = PolynomialFeatures(degree = 10)

X, y = generate_samples(10)
X_poly = poly.fit_transform(X)
model = StagewiseRegression()
model.feature_selection(X_poly, y, N=1000, eta=0.1)

plt.scatter(X, y)
plt.axis([-1, 1, -2, 2])
W = np.linspace(-1, 1, 100).reshape(100, 1)
W_poly = poly.fit_transform(W)
u = model.predict(W_poly)
plt.plot(W, u)
plt.show()
   













