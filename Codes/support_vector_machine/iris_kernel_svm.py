import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from machine_learning.support_vector_machine.lib.kernel_svm import KernelSVM

def rbf_kernel(x1, x2):
    sigma = 1.0
    return np.exp(-np.linalg.norm(x1 - x2, 2) ** 2 / sigma)
    
iris = datasets.load_iris()
X= iris["data"][:,(2,3)]
y = 2 * (iris["target"]==1).astype(np.int).reshape(-1,1) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
model = KernelSVM(kernel = rbf_kernel)
model.fit(X_train, y_train)

x0s = np.linspace(1, 7, 100)
x1s = np.linspace(0, 3, 100)
x0, x1 = np.meshgrid(x0s, x1s)
W = np.c_[x0.ravel(), x1.ravel()]
u= model.predict(W).reshape(x0.shape)
plt.plot(X_train[:, 0][y_train[:,0]==1] , X_train[:, 1][y_train[:,0]==1], "bs")
plt.plot(X_train[:, 0][y_train[:,0]==-1], X_train[:, 1][y_train[:,0]==-1], "yo")
plt.contourf(x0, x1, u, alpha=0.2)
plt.show()

























