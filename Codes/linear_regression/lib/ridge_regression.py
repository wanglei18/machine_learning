import numpy as np

class RidgeRegression:   
    def __init__(self, Lambda):
        self.Lambda = Lambda
        
    def fit(self, X, y):
        m, n = X.shape
        r = np.diag(self.Lambda * np.ones(n))  
        self.w = np.linalg.inv(X.T.dot(X) + r).dot(X.T).dot(y)
        return 
    
    def predict(self, X):
        return X.dot(self.w)


    








