import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#from machine_learning.dimension_reduction.pca.lib.pca import PCA
from sklearn.decomposition import PCA

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
X, Y = mnist.train.images, mnist.train.labels
model = PCA(n_components = 2)
Z = model.fit_transform(X)

plt.scatter(Z[:,0], Z[:,1], c = Y)
plt.show()








    
    






    
    
    










            
            
        
        
    



















