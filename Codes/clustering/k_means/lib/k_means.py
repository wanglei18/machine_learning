import numpy as np

class KMeans:
    def __init__(self, n_clusters = 1, max_iter = 50, random_state=0):
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
            centers = self.adjust_centers(assignments, X)
        return np.array(centers), np.array(assignments)
    
        



    
    






    
    
    










            
            
        
        
    



















