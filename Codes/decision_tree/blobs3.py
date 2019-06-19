import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from machine_learning.decision_tree.lib.decision_tree_classifier import DecisionTreeClassifier
from machine_learning.decision_tree.lib.random_forest_classifier_majority import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def convert_to_vector(y):
    m = len(y)
    k = np.max(y) + 1
    v = np.zeros(m * k).reshape(m,k)
    for i in range(m):
        v[i][y[i]] = 1
    return v

X, y = make_blobs(n_samples=1000, centers=3, random_state=0, cluster_std=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

plt.figure(0)
plt.axis([-3,4,-1,6])
plt.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c='b', marker='o', s=30)
plt.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='g', marker='^', s=30)
plt.scatter(X_train[:, 0][y_train==2], X_train[:, 1][y_train==2], c='y', marker='s', s=30)

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X_train, convert_to_vector(y_train))
y_pred = tree.predict(X_test)
print("decision tree accuracy= {}".format(accuracy_score(y_test, y_pred)))

plt.figure(1)
x0s = np.linspace(-3, 4, 100)
x1s = np.linspace(-1, 6, 100)
x0, x1 = np.meshgrid(x0s, x1s)
W = np.c_[x0.ravel(), x1.ravel()]
u= tree.predict(W).reshape(x0.shape)
plt.axis([-3,4,-1,6])
plt.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c='b', marker='o', s=30)
plt.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='g', marker='^', s=30)
plt.scatter(X_train[:, 0][y_train==2], X_train[:, 1][y_train==2], c='y', marker='s', s=30)
plt.contourf(x0, x1, u, c=u, alpha=0.2)

forest = RandomForestClassifier(max_depth=1, num_trees=100, feature_sample_rate=0.5, data_sample_rate=0.15)
forest.fit(X_train, convert_to_vector(y_train))
y_pred = forest.predict(X_test)
print("random forest accuracy= {}".format(accuracy_score(y_test, y_pred)))

plt.figure(2)
u= forest.predict(W).reshape(x0.shape)
plt.axis([-3,4,-1,6])
plt.scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], c='b', marker='o', s=30)
plt.scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], c='g', marker='^', s=30)
plt.scatter(X_train[:, 0][y_train==2], X_train[:, 1][y_train==2], c='y', marker='s', s=30)
plt.contourf(x0, x1, u, c=u, alpha=0.2)

plt.show()

























