import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import machine_learning.linear_regression.lib.linear_regression as lib

def process_features(X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        m, n = X.shape
        X = np.c_[np.ones((m,1)), X] 
        return X
        
housing = fetch_california_housing()
X = housing.data  
y = housing.target.reshape(-1,1)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = lib.LinearRegression()
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)
mse = lib.mean_squared_error(y_test, y_pred)
r2 = lib.r2_score(y_test, y_pred)
print("mse = {}, r2 = {}".format(mse, r2))










