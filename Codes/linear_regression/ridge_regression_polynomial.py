import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import machine_learning.linear_regression.lib.linear_regression as lib
from machine_learning.linear_regression.lib.ridge_regression import RidgeRegression

def generate_samples(m):
    X = 2 * (np.random.rand(m, 1) - 0.5) 
    y = X + np.random.normal(0, 0.3, (m,1))
    return X,y

np.random.seed(100)
poly = PolynomialFeatures(degree = 10)
X_train, y_train = generate_samples(30)
X_train = poly.fit_transform(X_train)
X_test, y_test = generate_samples(100)
X_test = poly.fit_transform(X_test)

Lambdas, train_r2s, test_r2s = [], [], []
for i in range(1, 200):
    Lambda = 0.01 * i
    Lambdas.append(Lambda)
    ridge = RidgeRegression(Lambda)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    train_r2s.append(lib.r2_score(y_train, y_train_pred))
    test_r2s.append(lib.r2_score(y_test, y_test_pred))
    
plt.figure(0)
plt.plot(Lambdas, train_r2s)
plt.figure(1)
plt.plot(Lambdas, test_r2s)
plt.show()
    








