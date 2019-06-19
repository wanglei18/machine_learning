import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from machine_learning.dimension_reduction.pca.lib.pca import PCA
from machine_learning.clustering.k_means.lib.k_means import KMeans

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
X, y = mnist.train.images, mnist.train.labels
model = PCA(n_components = 300)
Z = model.fit_transform(X)

clustering = KMeans(n_clusters=10, max_iter = 300)
centers, assignments = clustering.fit_transform(Z)
plt.figure(10)
plt.scatter(Z[:,0], Z[:,1], c = y)
plt.scatter(centers[:,0], centers[:,1], c='r', s=80)

centers_recovered = model.inverse_transform(centers)
for i in range(10):
    plt.figure(i)
    plt.imshow(centers_recovered[i].reshape(28,28))
plt.show()










    
    






    
    
    










            
            
        
        
    



















