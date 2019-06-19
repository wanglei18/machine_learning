import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

data = fetch_olivetti_faces()
images = data.images
plt.imshow(images[0])
plt.show()

data = images.reshape((len(data.images), -1))
n_pixels = data.shape[1]
X = data[:, :(n_pixels + 1) // 2]
y = data[:, n_pixels // 2:]












