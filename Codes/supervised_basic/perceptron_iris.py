import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from machine_learning.supervised_basic.perceptron import Perceptron
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X= iris["data"][:,(0,1)]
y = 2 * (iris["target"]==0).astype(np.int) - 1  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

plt.figure(1)
plt.axis([4,8,1,5])
plt.plot(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], "bs", ms=3)
plt.plot(X_train[:, 0][y_train==-1], X_train[:, 1][y_train==-1], "yo", ms=3)

model = Perceptron()
model.fit(X_train, y_train)

plt.figure(2)
plt.axis([4,8,1,5])
plt.plot(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1]+0.1, "bs", ms=3)
plt.plot(X_train[:, 0][y_train==-1], X_train[:, 1][y_train==-1]-0.1, "yo", ms=3)
x0 = np.linspace(4, 8, 200)
w = model.w
b = model.b
line = -w[0]/w[1] * x0 - b/w[1]
plt.plot(x0, line)

plt.figure(3)
plt.axis([4,8,1,5])
plt.plot(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], "bs", ms=3)
plt.plot(X_test[:, 0][y_test==-1], X_test[:, 1][y_test==-1], "yo", ms=3)
x0 = np.linspace(4, 8, 200)
line = -w[0]/w[1] * x0 - b/w[1]
plt.plot(x0, line)

plt.show()


















