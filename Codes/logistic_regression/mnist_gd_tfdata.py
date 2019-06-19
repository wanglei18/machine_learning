import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from machine_learning.logistic_regression.lib.logistic_regression_gd import LogisticRegression
import machine_learning.logistic_regression.lib.classification_metrics as metrics

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels
y_train = (y_train == 6).astype(np.int).reshape(-1,1)
y_test = (y_test == 6).astype(np.int).reshape(-1,1)

model = LogisticRegression()
model.fit(X_train, y_train, eta=0.01, N=3000)
proba = model.predict_proba(X_test)
entropy = metrics.cross_entropy(y_test, proba)
print("cross entropy = {}".format(entropy))



















