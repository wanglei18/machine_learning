import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import PythonCodes.dimension_reduction.pca.pca as pca



np.random.seed(0)        
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
model = pca.PCA(n_components = 1)
X_pca = model.fit_transform(X)
Ones = np.ones(X_pca.shape)

#plt.figure(10)
#plt.scatter(X[y==0, 0], X[y==0, 1], c="red")
#plt.scatter(X[y==1, 0], X[y==1, 1], c="blue")

plt.figure(12)
plt.scatter(X_pca[y==0, 0], Ones[y==0], c="red")
plt.scatter(X_pca[y==1, 0], 2*Ones[y==1], c="blue")

plt.show()



    
    






    
    
    










            
            
        
        
    



















