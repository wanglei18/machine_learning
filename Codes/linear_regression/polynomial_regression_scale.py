import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import machine_learning.linear_regression.lib.linear_regression as lib

def process_features(X):
    d = 2
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)
    m,n = X_poly.shape
    X_poly = np.c_[np.ones((m,1)), X_poly]  
    return X_poly

def generate_samples(m):
        sigma = 0.3
        miu = 1
        X = 2 * np.random.rand(m, 1) 
        y = 2 * X**2 - 4 * X + np.random.normal(miu, sigma, (m,1))
        return X,y

X_train, y_train = generate_samples(100)
plt.figure(0) 
plt.axis([0, 2, -4, 4])
plt.scatter(X_train, y_train)

plt.figure(1) 
plt.axis([0, 2, -4, 4])
plt.scatter(X_train, y_train)

X_train = process_features(X_train)
model = lib.LinearRegression()
model.fit(X_train, y_train)
print(model.w)

W = np.linspace(0, 2, 100).reshape(100, 1)
W_poly = process_features(W)
U = model.predict(W_poly)
plt.plot(W, U)

plt.show()




