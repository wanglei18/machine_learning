import numpy as np
import machine_learning.linear_regression.lib.linear_regression as lib

def generate_samples(m):
    X = 2*(np.random.rand(m, 1) - 0.5) 
    y = X + np.random.normal(0, 0.3, (m,1))
    return X,y

def process_features(X):
    m,n = X.shape
    X = np.c_[np.ones((m,1)), X]  
    return X

np.random.seed(0)
X_train, y_train = generate_samples(100)
X_train = process_features(X_train)
X_test, y_test = generate_samples(100)
X_test = process_features(X_test)

model = lib.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = lib.mean_squared_error(y_test, y_pred)
r2 = lib.r2_score(y_test, y_pred)
print("mse = {} and r2 = {}".format(mse, r2))
      
      