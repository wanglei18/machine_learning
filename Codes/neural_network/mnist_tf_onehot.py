import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
n_features = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_classes = 10
X = tf.placeholder(tf.float32, shape=(None, n_features))
y = tf.placeholder(tf.int64, shape=(None, n_classes))
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
outputs = tf.layers.dense(hidden2, n_classes)  
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=outputs)
loss = tf.reduce_mean(cross_entropy)  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)  
correct = tf.equal(tf.argmax(y,1), tf.argmax(outputs,1))
accuracy_score = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_test, y_test = mnist.test.images, mnist.test.labels
    for t in range(50000):
        i = np.random.randint(0, len(X_train))
        X_i = X_train[i].reshape(1,-1)
        y_i = y_train[i].reshape(1,-1)
        sess.run(train_op, feed_dict = {X : X_i, y : y_i})
    accuracy = accuracy_score.eval(feed_dict = {X : X_test, y : y_test})
    print("accuracy = {}".format(accuracy))

















