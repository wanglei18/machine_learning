import numpy as np
import heapq

class AgglomerativeClustering:
    def __init__(self, n_clusters = 1):
        self.k = n_clusters
            
    def fit_transform(self, X):
        m, n = X.shape
        C, centers = {}, {}
        assignments = np.zeros(m)
        for id in range(m):
            C[id] = [id]
            centers[id] = X[id]
            assignments[id] = id 
        H = []
        for i in range(m):
            for j in range(i+1, m):
                d = np.linalg.norm(X[i] - X[j], 2)
                heapq.heappush(H, (d, [i, j]))     
        new_id = m 
        while len(C) > self.k:
            distance, [id1, id2] = heapq.heappop(H)
            if id1 not in C or id2 not in C:
                continue
            C[new_id] = C[id1] + C[id2]
            for i in C[new_id]:
                assignments[i] = new_id
            del C[id1], C[id2], centers[id1], centers[id2]
            new_center = sum(X[C[new_id]]) / len(C[new_id])
            for id in centers:
                center = centers[id]
                d = np.linalg.norm(new_center - center, 2)
                heapq.heappush(H, (d, [id, new_id]))
            centers[new_id] = new_center
            new_id += 1
        return np.array(list(centers.values())), assignments
    



    
    






    
    
    










            
            
        
        
    



















