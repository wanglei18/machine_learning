import numpy as np
from copy import deepcopy

class DBSCAN:
    def __init__(self, eps = 0.5, min_sample = 5):
        self.eps = eps
        self.min_sample = min_sample
            
    def get_neighbors(self, X, i):
        if i in self.neighbors:
            return self.neighbors[i]
        m = len(X)
        distances = [np.linalg.norm(X[i] - X[j], 2) for j in range(m)]
        neighbors_i = [j for j in range(m) if distances[j] < self.eps]
        self.neighbors[i] = neighbors_i
        return neighbors_i
    
    def grow_cluster(self, X, i, neighbors_i, id):
        self.assignments[i] = id
        queue = deepcopy(neighbors_i)
        t = 0
        while t < len(queue):
            j = queue[t]
            if self.assignments[j] == 0:
                self.assignments[j] = id
                neighbors_j = self.get_neighbors(X, j)
                if len(neighbors_j) > self.min_sample:
                    queue += neighbors_j
            t += 1
                 
    def fit_transform(self, X):
        self.assignments = np.zeros(len(X))
        self.neighbors = {}
        id = 1
        for i in range(len(X)):
            if self.assignments[i] != 0:
                continue
            neighbors_i = self.get_neighbors(X, i)
            if len(neighbors_i) > self.min_sample:
                self.grow_cluster(X, i, neighbors_i, id)
                id += 1
        return self.assignments
    
           




    
    






    
    
    










            
            
        
        
    



















