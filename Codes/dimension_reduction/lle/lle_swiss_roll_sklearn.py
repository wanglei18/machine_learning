import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)
model = LocallyLinearEmbedding(n_components=2, n_neighbors=15)
Z = model.fit_transform(X)

plt.figure(19)
ax = plt.axes(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
ax.view_init(4, -72)


plt.figure(20)
plt.scatter(Z[:, 0], Z[:, 1], c=color)
plt.show()





















