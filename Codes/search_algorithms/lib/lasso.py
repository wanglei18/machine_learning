import numpy as np

class Lasso:
    def __init__(self, Lambda=1):
        self.Lambda = Lambda

    def fit(self, X, y, eta=0.1, N=1000):
        m,n = X.shape
        w = np.zeros((n,1)) 
        self.w = w
        for t in range(N):
            e = X.dot(w) - y
            v = 2 * X.T.dot(e) / m + self.Lambda * np.sign(w)
            w = w - eta * v  
            self.w += w
        self.w /= N
        return 
    
    def predict(self, X):
        return X.dot(self.w)





















