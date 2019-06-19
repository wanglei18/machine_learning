import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from machine_learning.logistic_regression.lib.softmax_regression_gd import SoftmaxRegression
from machine_learning.logistic_regression.lib.classification_metrics import accuracy_score
   
def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X] 
    return X
    
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = SoftmaxRegression()
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy = {}".format(accuracy))

















