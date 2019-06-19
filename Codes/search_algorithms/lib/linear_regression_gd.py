import numpy as np
    
class LinearRegression: 
    def fit(self, X, y, eta, epsilon):
        m, n = X.shape
        w = np.zeros((n,1)) 
        while True:
            e = X.dot(w) - y
            g = 2 * X.T.dot(e) / m  
            w = w - eta * g
            if np.linalg.norm(g, 2) < epsilon:
                break  
        self.w = w
    
    def predict(self, X):
        return X.dot(self.w)










