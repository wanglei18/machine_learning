import numpy as np

def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))
    
class LogisticRegression:
    def fit(self, X, y, N=1000):
        m, n = X.shape   
        w = np.zeros((n,1)) 
        for t in range(N):
            pred = sigmoid(X.dot(w))
            g = 1.0 / m * X.T.dot(pred - y)
            pred = pred.reshape(-1)
            D = np.diag(pred * (1 - pred))
            H = 1.0 / m * (X.T.dot(D)).dot(X)
            w = w - np.linalg.inv(H).dot(g)
        self.w = w
    
    def predict_proba(self, X):
        return sigmoid(X.dot(self.w))
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.int)
         



















