import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from machine_learning.logistic_regression.lib.softmax_regression_sgd import SoftmaxRegression
from machine_learning.logistic_regression.lib.classification_metrics import accuracy_score

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels

model = SoftmaxRegression()
model.fit(X_train, y_train, eta_0=50, eta_1=100, N=100000)
proba = model.predict_proba(X_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), 
                          np.argmax(proba, axis=1))
print("accuracy = {}".format(accuracy))


















