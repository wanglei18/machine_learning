import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt

def generate_samples(m):
    X = 2 * (np.random.rand(m, 1) - 0.5) 
    y = X + np.random.normal(0, 0.3, (m,1))
    return X, y

np.random.seed(100)
X, y = generate_samples(10)
poly = PolynomialFeatures(degree = 10)
X_poly = poly.fit_transform(X)
W = np.linspace(-1, 1, 100).reshape(100, 1)
W_poly = poly.fit_transform(W)

plt.figure(0) 
plt.axis([-1, 1, -2, 2])
plt.scatter(X, y)
model = linear_model.LinearRegression()
model.fit(X_poly, y)
U = model.predict(W_poly)
plt.plot(W, U)

plt.figure(1) 
plt.axis([-1, 1, -2, 2])
plt.scatter(X, y)
model = linear_model.Lasso(alpha=0.001)
model.fit(X_poly, y)
U = model.predict(W_poly)
plt.plot(W, U)

plt.figure(2) 
plt.axis([-1, 1, -2, 2])
plt.scatter(X, y)
model = linear_model.Lasso(alpha=0.01)
model.fit(X_poly, y)
U = model.predict(W_poly)
plt.plot(W, U)

plt.figure(3) 
plt.axis([-1, 1, -2, 2])
plt.scatter(X, y)
model = linear_model.Lasso(alpha=0.1)
model.fit(X_poly, y)
U = model.predict(W_poly)
plt.plot(W, U)

plt.figure(4) 
plt.axis([-1, 1, -2, 2])
plt.scatter(X, y)
model = linear_model.Lasso(alpha=0.2)
model.fit(X_poly, y)
U = model.predict(W_poly)
plt.plot(W, U)

plt.figure(5) 
plt.axis([-1, 1, -2, 2])
plt.scatter(X, y)
model = linear_model.Lasso(alpha=0.4)
model.fit(X_poly, y)
U = model.predict(W_poly)
plt.plot(W, U)

plt.show()

   







