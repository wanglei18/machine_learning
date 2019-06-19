import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from machine_learning.dimension_reduction.pca.lib.kernel_pca import KernelPCA
from machine_learning.dimension_reduction.pca.lib.pca import PCA
from machine_learning.dimension_reduction.lda.lib.lda import LDA

def rbf_kernel(x1, x2):
    gamma = 15
    return np.exp(-gamma * np.linalg.norm(x1-x2, 2) ** 2)

np.random.seed(0)        
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
pca = PCA(n_components = 1)
X_pca = pca.fit_transform(X).reshape(-1)
kpca = KernelPCA(n_components = 1, kernel = rbf_kernel)
X_kpca = kpca.fit_transform(X).reshape(-1)
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X, y).reshape(-1)

plt.figure(1)
plt.scatter(X_pca, np.ones(X_pca.shape), c=y, cmap='rainbow')
plt.figure(2)
plt.scatter(X_kpca, np.ones(X_kpca.shape), c=y, cmap='rainbow')
plt.figure(3)
plt.scatter(X_lda, np.ones(X_lda.shape), c=y, cmap='rainbow')
plt.show()




    
    






    
    
    










            
            
        
        
    



















