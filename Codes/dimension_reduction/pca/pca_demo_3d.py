import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PythonCodes.dimension_reduction.pca as pca

np.random.seed(0)        

def f(x0,x1):
    #return -x0 - x1  + 0.6
    return x0 + x1

def generate_samples(m):
    x = np.random.rand(m,3)
    for i in range(m):
        x[i][2] = f(x[i][0], x[i][1]) + np.random.normal(0, 0.1)
    return x

x = generate_samples(100)
model = pca.PCA(n_components = 2)
z = model.fit_transform(x)

plt.figure(1)
ax = plt.axes(projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], "bs")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
x0 = np.linspace(0, 1, 100)
x1 = np.linspace(0, 1, 100)
X0, X1 = np.meshgrid(x0, x1)
X2 = f(X0, X1)
ax.plot_surface(X0, X1, X2, alpha=0.5, color='w')

plt.figure(2)
plt.scatter(z[:,0], z[:,1])

plt.show()



    
    






    
    
    










            
            
        
        
    



















