import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from machine_learning.dimension_reduction.pca.lib.kernel_pca import KernelPCA
from machine_learning.dimension_reduction.pca.lib.pca import PCA

def rbf_kernel(x1, x2):
    gamma = 15
    return np.exp(-gamma * np.linalg.norm(x1 - x2, 2) ** 2)

np.random.seed(0)        
X, y = make_moons(n_samples=500, noise=0.01)
plt.figure(0)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

pca = PCA(n_components = 1)
X_pca = pca.fit_transform(X).reshape(-1)
kpca = KernelPCA(n_components = 1, kernel = rbf_kernel)
X_kpca = kpca.fit_transform(X).reshape(-1)

plt.figure(1)
plt.scatter(X_pca, np.ones(X_pca.shape), c=y, cmap='rainbow')
plt.figure(2)
plt.scatter(X_kpca, np.ones(X_kpca.shape), c=y, cmap='rainbow')
plt.show()



    
    






    
    
    










            
            
        
        
    



















