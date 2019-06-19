import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from machine_learning.logistic_regression.lib.softmax_regression_gd import SoftmaxRegression
from machine_learning.logistic_regression.lib.classification_metrics import accuracy_score
   
def convert_to_vectors(c):
    m = len(c)
    k = np.max(c) + 1
    y = np.zeros(m * k).reshape(m,k)
    for i in range(m):
        y[i][c[i]] = 1
    return y
    
def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X] 
    return X
    
iris = datasets.load_iris()
X = iris["data"]
c = iris["target"]
y = convert_to_vectors(c)
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.2)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = SoftmaxRegression()
model.fit(X_train, y_train)
c_pred = model.predict(X_test)
accuracy = accuracy_score(c_test, c_pred)
print("accuracy = {}".format(accuracy))

















