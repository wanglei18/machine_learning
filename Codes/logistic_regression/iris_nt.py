import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from machine_learning.logistic_regression.lib.logistic_regression_nt import LogisticRegression
import machine_learning.logistic_regression.lib.classification_metrics as metrics
  
def process_features(X):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X] 
    return X
    
iris = datasets.load_iris()
X = iris["data"]
y = (iris["target"]==2).astype(np.int).reshape(-1,1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = LogisticRegression()
model.fit(X_train, y_train, N=1000)
proba = model.predict_proba(X_test)
entropy = metrics.cross_entropy(y_test, proba)
print("cross entropy = {}".format(entropy))


















