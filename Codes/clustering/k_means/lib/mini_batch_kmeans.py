import numpy as np

class MiniBatchKMeans:
    def __init__(self, n_clusters = 1, max_iter = 50, batch_size=1, random_state=0):
        self.k = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        np.random.seed(random_state)
         
    def assign_to_centers(self, centers, assignments, X, batch):
        for i in batch:
            distances = [np.linalg.norm(X[i] - centers[j], 2) for j in range(self.k)] 
            j = np.argmin(distances)
            assignments[i] = j
        return assignments 
    
    def adjust_centers(self, centers, assignments, counts, X, batch):
        for i in batch:
            j = int(assignments[i])
            counts[j] += 1
            eta = 1.0 / counts[j]
            centers[j] = (1- eta) * centers[j] + eta * X[i]
        return
               
    def fit_transform(self, X):
        idx = np.random.randint(0, len(X), self.k)
        centers = np.array([X[i] for i in idx])
        assignments = np.zeros(len(X))
        counts = np.zeros(self.k)
        for iter in range(self.max_iter):
            batch = np.random.randint(0, len(X), self.batch_size)
            self.assign_to_centers(centers, assignments, X, batch)
            self.adjust_centers(centers, assignments, counts, X, batch)
        batch = np.random.randint(0, len(X), len(X))
        self.assign_to_centers(centers, assignments, X, batch)
        return np.array(centers), np.array(assignments)
    
        



    
    






    
    
    










            
            
        
        
    



















