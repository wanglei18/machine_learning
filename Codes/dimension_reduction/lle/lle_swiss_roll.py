import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import PythonCodes.dimension_reduction.lle.lle as lle

np.random.seed(0)
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)
model = lle.LLE(n_components=2, n_neighbors=12)
Z = model.fit_transform(X)

plt.figure(20)
plt.scatter(Z[:, 0], Z[:, 1], c=color)
plt.show()





















