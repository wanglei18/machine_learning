import numpy as np

def softmax(scores):
    e = np.exp(scores)
    s = e.sum(axis=1)
    for i in range(len(s)):
        e[i] /= s[i]
    return e
    
class SoftmaxRegression:
    def fit(self, X, y, eta_0=50, eta_1=100, N=1000):
        m, n = X.shape
        m, k = y.shape
        w = np.zeros(n * k).reshape(n,k) 
        self.w = w
        for t in range(N):
            i = np.random.randint(m)
            x = X[i].reshape(1,-1)
            proba = softmax(x.dot(w))
            g = x.T.dot(proba - y[i])
            w = w - eta_0 / (t + eta_1) * g 
            self.w += w
        self.w /= N
    
    def predict_proba(self, X):
        return softmax(X.dot(self.w))
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    


















