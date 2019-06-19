import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import PythonCodes.dimension_reduction.pca.kernel_pca as kpca

def rbf_kernel(x1, x2):
    gamma = 10
    return np.exp(-gamma * np.linalg.norm(x1-x2, 2) ** 2)

np.random.seed(0)        
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
model = kpca.KernelPCA(n_components = 2, kernel = rbf_kernel)
X_kpca = model.fit_transform(X)
Ones = np.ones(X_kpca.shape)

plt.figure(10)
plt.scatter(X[y==0, 0], X[y==0,1], c="red")
plt.scatter(X[y==1, 0], X[y==1,1], c="blue")

plt.figure(12)
plt.scatter(X_kpca[y==0, 0], X_kpca[y==0,1], c="red")
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1,1], c="blue")

plt.show()



    
    






    
    
    










            
            
        
        
    



















