import numpy as np

class StagewiseRegression:   
    def feature_selection(self, X, y, N, eta):
        m, n = X.shape
        norms = np.linalg.norm(X, 2, axis=0).reshape(-1, 1)
        w = np.zeros(n)
        t = 0
        r = y
        while t < N:
            c = X.T.dot(r) / norms
            j_max = np.argmax(abs(c))
            delta = eta * np.sign(c[j_max])
            w[j_max] = w[j_max] + delta 
            r = r - delta * X[:,j_max].reshape(-1,1)
            t = t + 1 
        self.w = w
        return w
    
    def predict(self, X):
        return X.dot(self.w)
    

     













