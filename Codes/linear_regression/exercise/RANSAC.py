import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

def generate_samples(m, k):
    X_normal = 2 * (np.random.rand(m, 1) - 0.5) 
    y_normal = X_normal + np.random.normal(0, 0.1, (m,1))
    X_outlier = 2 * (np.random.rand(k, 1) - 0.5)
    y_outlier = X_outlier + np.random.normal(3, 0.1, (k,1))
    X = np.concatenate((X_normal, X_outlier), axis=0)
    y = np.concatenate((y_normal, y_outlier), axis=0)
    return X, y

np.random.seed(0)
X, y = generate_samples(100, 5)
model = LinearRegression()
model.fit(X, y)
plt.figure(0)
plt.scatter(X, y)
W = np.linspace(-1, 1, 100).reshape(100, 1)
u = model.predict(W)
plt.plot(W, u)

model = RANSACRegressor()
model.fit(X, y)
plt.figure(1)
plt.scatter(X, y)
W = np.linspace(-1, 1, 100).reshape(100, 1)
u = model.predict(W)
plt.plot(W, u)


plt.show()




















