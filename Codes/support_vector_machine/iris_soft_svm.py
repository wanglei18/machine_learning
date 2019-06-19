import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from machine_learning.support_vector_machine.lib.soft_svm_smo import SoftSVM
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
 
iris = datasets.load_iris()
X = iris["data"][:,(2,3)]
y = 2 * (iris["target"]==2).astype(np.int).reshape(-1,1) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

model = SoftSVM(C=5.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy= {}".format(accuracy))

plt.figure(2)
plt.axis([1,7,0,3])
plt.plot(X_train[:, 0][y_train[:,0]==1] , X_train[:, 1][y_train[:,0]==1], "bs",  ms=3)
plt.plot(X_train[:, 0][y_train[:,0]==-1], X_train[:, 1][y_train[:,0]==-1], "yo",  ms=3)
x0 = np.linspace(1, 7, 200)
w = model.w
b = model.b
line = -w[0]/w[1] * x0 - b/w[1]
plt.plot(x0, line, color='black')
plt.show()























