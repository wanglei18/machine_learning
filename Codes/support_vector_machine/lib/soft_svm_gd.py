import numpy as np


class SoftSVM:
    def __init__(self, C = 1000):
        self.C = C  
    
    def fit(self, X, y, eta=0.01, N=5000):
        m, n = X.shape        
        w, b = np.zeros((n,1)), 0 
        for r in range(N):
            s = (X.dot(w) + b) * y
            e = (s < 1).astype(np.int).reshape(-1,1)
            g_w = 1 / (self.C * m ) * w - 1 / m * X.T.dot(y * e)
            g_b = - 1 / m * (y * e).sum()
            w = w - eta * g_w
            b = b - eta * g_b
        self.w = w
        self.b = b
    
    def predict(self, X):
        return np.sign(X.dot(self.w)+self.b)
















