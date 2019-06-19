import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = fetch_olivetti_faces()
data = data.images.reshape((len(data.images), -1))
n_pixels = data.shape[1]
X = data[:, :(n_pixels + 1) // 2]
y = data[:, n_pixels // 2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 

image_shape = (64, 64)
id = 5
true_face = np.hstack((X_test[id], y_test[id]))
pred_face = np.hstack((X_test[id], y_pred[id]))

plt.figure(0)
plt.imshow(true_face.reshape(image_shape), interpolation="nearest")
plt.figure(1)
plt.imshow(pred_face.reshape(image_shape), interpolation="nearest")
plt.show()










