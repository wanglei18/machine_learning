import numpy as np

class Lasso:  
    def __init__(self, Lambda=1):
        self.lambda_ = Lambda
    
    def soft_threshold(self, t, x):
        if x>t:
            return x-t
        elif x>=-t:
            return 0
        else:
            return x+t

    def fit(self, X, y, N=1000):
        m,n = X.shape
        alpha = 2 * np.sum(X**2, axis=0) / m
        w = np.zeros(n)
        for t in range(N):
            j = t % n
            w[j]=0
            e_j = X.dot(w.reshape(-1,1)) - y
            beta_j = 2 * X[:, j].dot(e_j) / m
            u = self.soft_threshold(self.lambda_/alpha[j], -beta_j/alpha[j])
            w[j] = u
        self.w = w
    
    def predict(self, X):
        return X.dot(self.w.reshape(-1,1))





















