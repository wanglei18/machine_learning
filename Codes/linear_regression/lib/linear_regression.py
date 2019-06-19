import numpy as np

class LinearRegression:
    def fit(self, X, y):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return 
    
    def predict(self, X):
        return X.dot(self.w)

def mean_squared_error(y_true, y_pred):
    return np.average((y_true - y_pred)**2, axis=0)
    
def r2_score(y_true, y_pred):
    numerator = (y_true - y_pred)**2
    denominator = (y_true - np.average(y_true, axis=0))**2
    return 1- numerator.sum(axis=0) / denominator.sum(axis=0)





