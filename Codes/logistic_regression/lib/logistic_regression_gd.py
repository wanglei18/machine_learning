import numpy as np

def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))
    
def get_cross_entropy(y_true, y_pred):
    return np.average(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    
class LogisticRegression:
    def fit(self, X, y, eta=0.1, N = 1000):
        m, n = X.shape   
        w = np.zeros((n,1)) 
        for t in range(N):
            h = sigmoid(X.dot(w))
            g = 1.0 / m * X.T.dot(h - y)
            w = w - eta * g
        self.w = w

    def predict_proba(self, X):
        return sigmoid(X.dot(self.w))
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.int)
             





















