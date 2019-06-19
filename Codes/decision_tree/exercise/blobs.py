import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, centers=3, random_state=0, cluster_std=0.4)
plt.figure(0)
plt.scatter(X[:, 0][y==0], X[:, 1][y==0], c='b', marker='o', s=10)
plt.scatter(X[:, 0][y==1], X[:, 1][y==1], c='g', marker='^', s=10)
plt.scatter(X[:, 0][y==2], X[:, 1][y==2], c='y', marker='s', s=10)
plt.show()

























