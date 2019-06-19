import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from machine_learning.dimension_reduction.pca.lib.pca import PCA

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
X, Y = mnist.train.images, mnist.train.labels
model = PCA(n_components = 100)
Z = model.fit_transform(X)
X_recovered = model.inverse_transform(Z).astype(int)

plt.figure(0)
plt.imshow(X[0].reshape(28,28))
plt.figure(1)
plt.imshow(X_recovered[0].reshape(28,28))
plt.figure(2)
plt.scatter(Z[:,0], Z[:,1], c = Y)
plt.show()









    
    






    
    
    










            
            
        
        
    



















