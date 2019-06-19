import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters = 1, max_iter = 6, random_state=0):
        self.k = n_clusters
        self.max_iter = max_iter
        np.random.seed(random_state)
         
    def assign_to_centers(self, centers, X):
        assignments = []
        for i in range(len(X)):
            distances = [np.linalg.norm(X[i] - centers[j], 2) for j in range(self.k)] 
            assignments.append(np.argmin(distances))
        return assignments 
    
    def adjust_centers(self, assignments, X):
        new_centers = []
        for j in range(self.k):
            cluster_j = [X[i] for i in range(len(X)) if assignments[i] == j]
            new_centers.append(np.mean(cluster_j, axis=0))
        return new_centers  
            
    def fit_transform(self, X):
        idx = np.random.randint(0, len(X), self.k)
        centers = [X[i] for i in idx]
        for iter in range(self.max_iter):
            assignments = self.assign_to_centers(centers, X)
            plt.figure(iter)
            plt.scatter(X[:, 0], X[:, 1], c = assignments)
            plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], c='r', s=80)
            centers = self.adjust_centers(assignments, X)
            
        plt.show()
        
        return centers, assignments
    
X, y = make_blobs(n_samples=300, centers=3, random_state=0, cluster_std=0.8)   
model = KMeans(n_clusters=3)
model.fit_transform(X)  



    
    






    
    
    










            
            
        
        
    



















