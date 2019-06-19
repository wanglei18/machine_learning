import numpy as np

def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))
    
class LogisticRegression:
    def fit(self, X, y, eta_0=10, eta_1=50, N=1000):
        m, n = X.shape 
        w = np.zeros((n,1)) 
        self.w = w
        for t in range(N):
            i = np.random.randint(m)
            x = X[i].reshape(1,-1)
            pred = sigmoid(x.dot(w))
            g = x.T * (pred - y[i])
            w = w - eta_0 / (t + eta_1) * g  
            self.w += w
        self.w /= N
    
    def predict_proba(self, X):
        return sigmoid(X.dot(self.w))
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.int)
         


















