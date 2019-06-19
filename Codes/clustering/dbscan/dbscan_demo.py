import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class DBSCAN:
    def __init__(self, eps = 0.5, min_sample = 5):
        self.eps = eps
        self.min_sample = min_sample
            
    def get_neighbors(self, X, i):
        m = len(X)
        distances = [np.linalg.norm(X[i] - X[j], 2) for j in range(m)]
        neighbors_i = [j for j in range(m) if distances[j] < self.eps and j != i]
        return neighbors_i
             
    def fit_transform(self, X):
        assignments = np.zeros(len(X))
        
        plt.figure(-1)
        plt.scatter(X[:,0], X[:,1], c = assignments)
        plt.show()
        
        assignments[0] = 1
        plt.figure(0)
        plt.scatter(X[:,0], X[:,1], c = assignments)
        plt.show()
        
        n0 = self.get_neighbors(X, 0)
        n1 = []
        for j in n0:
            assignments[j] = 1
            n1 += self.get_neighbors(X, j)
        plt.figure(1)
        plt.scatter(X[:,0], X[:,1], c = assignments)
        plt.show()
        
        n2= []
        for j in n1:
            assignments[j] = 1
            n2 += self.get_neighbors(X, j)
        plt.figure(2)
        plt.scatter(X[:,0], X[:,1], c = assignments)
        plt.show()
        
        n3= []
        for j in n2:
            assignments[j] = 1
            n3 += self.get_neighbors(X, j)
        plt.figure(3)
        plt.scatter(X[:,0], X[:,1], c = assignments)
        plt.show()
        
        n4= []
        for j in n3:
            assignments[j] = 1
            n4 += self.get_neighbors(X, j)
        plt.figure(4)
        plt.scatter(X[:,0], X[:,1], c = assignments)
        plt.show()
        
        
        
        
        
    
X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.5)     
model = DBSCAN(eps = 0.5, min_sample = 2)
model.fit_transform(X)
plt.show()



    
    






    
    
    










            
            
        
        
    



















