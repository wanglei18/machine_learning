import numpy as np

def softmax(scores):
    e = np.exp(scores)
    s = e.sum(axis=1)
    for i in range(len(s)):
        e[i] /= s[i]
    return e
    
class SoftmaxRegression:
    def fit(self, X, y, eta=0.1, N=5000):
        m, n = X.shape
        m, k = y.shape
        w = np.zeros(n * k).reshape(n,k) 
        for t in range(N):
            proba = softmax(X.dot(w))
            g = X.T.dot(proba - y) / m
            w = w - eta * g 
        self.w = w
    
    def predict_proba(self, X):
        return softmax(X.dot(self.w))
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
                        

        


















