import numpy as np
import matplotlib.pyplot as plt
from machine_learning.clustering.k_means.lib.k_means import KMeans
from machine_learning.clustering.hierarchical_clustering.agglomerative_clustering import AgglomerativeClustering
 
def generate_ball(x, radius, m): 
    r = radius * np.random.rand(m)
    pi = 3.14
    theta = 2 * pi * np.random.rand(m)
    B = np.zeros((m,2))
    for i in range(m):
        B[i][0] = x[0] + r[i] * np.cos(theta[i])
        B[i][1] = x[1] + r[i] * np.sin(theta[i])
    return B

B1 = generate_ball([0,0], 1, 100)
B2 = generate_ball([0,2], 1, 100)
B3 = generate_ball([5,1], 0.5, 10)
X = np.concatenate((B1, B2, B3), axis=0)

kmeans = KMeans(n_clusters = 2)
km_centers, km_assignments = np.array(kmeans.fit_transform(X))

agg = AgglomerativeClustering(n_clusters = 2)
agg_centers, agg_assignments = agg.fit_transform(X)

plt.figure(7)
plt.axis([-2, 6, -2, 4])
plt.scatter(X[:,0], X[:,1], c='y')

plt.figure(8)
plt.axis([-2, 6, -2, 4])
plt.scatter(X[:,0], X[:,1], c='y')
plt.scatter(km_centers[:,0], km_centers[:,1], c='b', marker='*', s=300)

plt.figure(9)
plt.axis([-2, 6, -2, 4])
plt.scatter(X[:,0], X[:,1], c='y')
plt.scatter(agg_centers[:,0], agg_centers[:,1], c='b', marker='*', s=300)
plt.show()





    
    






    
    
    










            
            
        
        
    



















