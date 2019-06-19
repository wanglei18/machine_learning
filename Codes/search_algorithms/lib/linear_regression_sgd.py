import numpy as np
    
class LinearRegression:    
    def fit(self, X, y, eta_0=10, eta_1=50, N=3000):
        m, n = X.shape
        w = np.zeros((n,1))
        self.w = w
        for t in range(N):
            i = np.random.randint(m)
            x = X[i].reshape(1,-1)
            e = x.dot(w) - y[i]
            g = 2 * e * x.T
            w = w - eta_0 * g / (t + eta_1) 
            self.w += w
        self.w /= N
    
    def predict(self, X):
        return X.dot(self.w)












