import numpy as np
    
class LinearRegression:    
    def fit(self, X, y, eta_0=10, eta_1=50, N=3000, B=10):
        m, n = X.shape
        w = np.zeros((n,1))
        self.w = w
        for t in range(N):
            batch = np.random.randint(low=0, high=m, size=B)
            X_batch = X[batch].reshape(B,-1)
            y_batch = y[batch].reshape(B,-1)
            e = X_batch.dot(w) - y_batch
            g = 2 * X_batch.T.dot(e) / B
            w = w - eta_0 * g / (t + eta_1) 
            self.w += w
        self.w /= N
    
    def predict(self, X):
        return X.dot(self.w)












