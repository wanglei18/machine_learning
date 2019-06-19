import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from machine_learning.clustering.k_means.lib.k_means import KMeans 

X, y = make_blobs(n_samples=300, centers=3,
                  random_state=0, cluster_std=0.8)
plt.figure(-1)
plt.scatter(X[:, 0], X[:, 1])
model = KMeans(n_clusters = 3, max_iter = 100)
centers, assignments = model.fit_transform(X)

plt.figure(0)
plt.scatter(X[:, 0], X[:, 1], c = assignments)
plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], c='r', s=80)
plt.show()

    






    
    
    










            
            
        
        
    



















