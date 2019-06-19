import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from machine_learning.dimension_reduction.pca.lib.kernel_pca import KernelPCA
from machine_learning.dimension_reduction.pca.lib.pca import PCA

def rbf_kernel(x1, x2):
    gamma = 15
    return np.exp(-gamma * np.linalg.norm(x1 - x2, 2) ** 2)

np.random.seed(0)        
X, y = make_moons(n_samples=200, noise=0.01)
plt.figure(0)
plt.scatter(X[:, 0][y==0], X[:, 1][y==0], c='b', marker='o', s=10)
plt.scatter(X[:, 0][y==1], X[:, 1][y==1], c='y', marker='s', s=10)

pca = PCA(n_components = 1)
X_pca = pca.fit_transform(X).reshape(-1)
kpca = KernelPCA(n_components = 1, kernel = rbf_kernel)
X_kpca = kpca.fit_transform(X).reshape(-1)

plt.figure(1)
plt.scatter(X_pca[y==0], np.ones(X_pca[y==0].shape), c='b', marker='o', s=5)
plt.scatter(X_pca[y==1], np.ones(X_pca[y==1].shape), c='y', marker='s', s=5)
plt.figure(2)
plt.scatter(X_kpca[y==0], np.ones(X_kpca[y==0].shape), c='b', marker='o', s=5)
plt.scatter(X_kpca[y==1], np.ones(X_kpca[y==1].shape), c='y', marker='s', s=5)
plt.show()



    
    






    
    
    










            
            
        
        
    



















