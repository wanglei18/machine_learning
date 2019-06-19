import numpy as np

class Perceptron:
    def fit(self, X, y):
        m, n = X.shape
        w = np.zeros((n,1)) 
        b = 0
        done = False
        while not done:
            done = True
            for i in range(m):
                x = X[i].reshape(1,-1)
                if y[i] * (x.dot(w) + b) <= 0:
                    w = w + y[i] * x.T
                    b = b + y[i]
                    done = False  
        self.w = w
        self.b = b
    
    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)
        


















