import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2):
    gamma = 5
    return np.exp(-gamma * np.linalg.norm(x1-x2, 2) ** 2)

class MeanShift:
    def __init__(self, n_clusters = 1, bandwidth=3, max_iter = 50, random_state=0):
        self.k = n_clusters
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        np.random.seed(random_state)
       
    def get_neighbors(self, X, i):
        m = len(X)
        distances = [np.linalg.norm(X[i] - X[j], 2) for j in range(m)]
        neighbors_i = [j for j in range(m) if distances[j] < self.bandwidth]
        return neighbors_i
        
    def fit_transform(self, X):
        m, n = X.shape
        for t in range(self.max_iter):
            Z = np.zeros(X.shape)
            done = True
            for i in range(m):
                neighbors_i = self.get_neighbors(X, i)
                sum = 0
                for j in neighbors_i:
                    d = rbf_kernel(X[i], X[j])
                    Z[i] += d * X[j]
                    sum += d
                Z[i] /= sum
                if np.linalg.norm(X[i] - Z[i]) > 0.1:
                    done = False
            for i in range(m):
                for j in range(n):
                    X[i][j] = Z[i][j]
            plt.figure(t+1)
            plt.axis([-3, 3, 0, 6])
            plt.scatter(X[:, 0], X[:, 1]) 
            if done: 
                break
        return X

X, y = make_blobs(n_samples=100, centers=3,
                  random_state=0, cluster_std=0.4)
plt.figure(0)
plt.axis([-3, 3, 0, 6])
plt.scatter(X[:, 0], X[:, 1])
model = MeanShift(n_clusters = 3, bandwidth=5, max_iter = 1000)
X = model.fit_transform(X)
plt.show()

   
                
            
        
        
        
        
    
        



    
    






    
    
    










            
            
        
        
    



















