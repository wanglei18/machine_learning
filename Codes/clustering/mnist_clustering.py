import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from machine_learning.dimension_reduction.pca.lib.pca import PCA
from machine_learning.clustering.hierarchical_clustering.agglomerative_clustering import AgglomerativeClustering
from machine_learning.clustering.k_means.lib.k_means import KMeans 
from machine_learning.clustering.k_means.lib.mini_batch_kmeans import MiniBatchKMeans

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
X, y = mnist.test.images, mnist.test.labels
#model = KMeans(n_clusters=10, max_iter=300)
model = MiniBatchKMeans(n_clusters=10, max_iter=1000, batch_size=30)
centers, assignments = model.fit_transform(X)

for i in range(10):
    plt.figure(i)
    plt.imshow(centers[i].reshape(28,28))
plt.show()










    
    






    
    
    










            
            
        
        
    



















