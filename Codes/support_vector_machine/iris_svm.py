import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from machine_learning.support_vector_machine.lib.svm_smo import SVM

def plot_figure(X, y, model):
    z = np.linspace(4, 8, 200)
    w = model.w
    b = model.b
    L = - w[0] / w[1] * z - b / w[1]
    plt.plot(X[:, 0][y[:, 0]==1], X[:, 1][y[:, 0]==1], "bs",  ms=3)
    plt.plot(X[:, 0][y[:, 0]==-1], X[:, 1][y[:, 0]==-1], "yo",  ms=3)
    plt.plot(z, L)
    plt.show()

iris = datasets.load_iris()
X= iris["data"][:, (0,1)]
y = 2 * (iris["target"]==0).astype(np.int).reshape(-1,1) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

model = SVM()
model.fit(X_train, y_train, N=10)
plot_figure(X_train, y_train, model)
plot_figure(X_test, y_test, model)



















