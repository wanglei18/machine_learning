import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from machine_learning.logistic_regression.lib.logistic_regression_sgd import LogisticRegression
import machine_learning.logistic_regression.lib.classification_metrics as metrics

def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X] 
    return X

mnist = fetch_mldata('MNIST original', data_home="./")
m, n = mnist["data"].shape
X = mnist["data"]
y = (mnist["target"] == 6).astype(np.int).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = LogisticRegression()
model.fit(X_train, y_train, eta_0=10, eta_1=50, N=3000)
proba = model.predict_proba(X_test)
entropy = metrics.cross_entropy(y_test, proba)
print("cross entropy = {}".format(entropy))



















