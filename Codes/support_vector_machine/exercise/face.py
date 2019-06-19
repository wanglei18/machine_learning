import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print(lfw_people.target_names)
X = lfw_people.data
y = 2 * (lfw_people.target==3).astype(np.int).reshape(-1,1) - 1

m, h, w = lfw_people.images.shape
print(h,w)
plt.imshow(X[2].reshape(h, w))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = SVC(kernel='rbf', C=1, gamma=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy = {}".format(accuracy))


























