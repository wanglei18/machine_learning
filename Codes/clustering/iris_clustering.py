import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from machine_learning.clustering.k_means.lib.k_means import KMeans
from machine_learning.clustering.hierarchical_clustering.agglomerative_clustering2 import AgglomerativeClustering
   
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"] 

model = AgglomerativeClustering(n_clusters=3)
centers, assignments = model.fit_transform(X)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

plt.figure(0)
plt.scatter(Z[:,0], Z[:,1], c = y)
plt.figure(1)
plt.scatter(Z[:,0], Z[:,1], c = assignments)
plt.show()




















