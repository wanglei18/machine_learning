import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=1000, noise=0.1)
plt.scatter(X[:, 0][y==0], X[:, 1][y==0], c='b', marker='o', s=10)
plt.scatter(X[:, 0][y==1], X[:, 1][y==1], c='y', marker='^', s=10)

plt.show()

























