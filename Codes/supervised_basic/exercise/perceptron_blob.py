from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
  
X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=0.6, random_state=0)

plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", ms=3)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", ms=3)
plt.show()
















