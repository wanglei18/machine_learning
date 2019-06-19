import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
    
class LinearRegressionSGD:    
    def fit(self, X, y, eta_0=10, eta_1=50, N=3000):
        m, n = X.shape
        w = np.zeros((n,1))
        self.w = w
        W = np.zeros((N,2))
        for t in range(N):
            W[t][0] = w[0]
            W[t][1] = w[1]
            i = np.random.randint(m)
            x = X[i].reshape(1,-1)
            e = x.dot(w) - y[i]
            gradient = 2 * e * x.T
            w = w - eta_0 * gradient / (t + eta_1) 
            self.w += w
        self.w /= N
        plt.figure(0)
        plt.scatter(W[:,0], W[:,1], s=15)
        plt.plot(W[:,0], W[:,1])
    
    def predict(self, X):
        return X.dot(self.w)
    
class LinearRegression: 
    def fit(self, X, y, eta, N=1000):
        m, n = X.shape
        w = np.zeros((n,1))
        W = np.zeros((N,2)) 
        for t in range(N):
            W[t][0] = w[0]
            W[t][1] = w[1]
            e = X.dot(w) - y
            g = 2 * X.T.dot(e) / m  
            w = w - eta * g 
        self.w = w
        plt.figure(1)
        plt.scatter(W[:,0], W[:,1], s=15)
        plt.plot(W[:,0], W[:,1])
    
    def predict(self, X):
        return X.dot(self.w)
    
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, bias=0, random_state=0)
y = y.reshape(-1,1)

model = LinearRegression()
model.fit(X,y,eta=0.01, N=1000)
print(model.w)

model = LinearRegressionSGD()
model.fit(X,y)
print(model.w)

plt.show()










