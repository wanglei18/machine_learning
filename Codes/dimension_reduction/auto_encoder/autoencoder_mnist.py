import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

n_features = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_encoder_outputs = 2
n_hidden3 = n_hidden2
n_hidden4 = n_hidden1
n_decoder_ouputs = n_features 

X = tf.placeholder(tf.float32, shape=(None, n_features))
hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu)
encoder_outputs = tf.layers.dense(hidden2, n_encoder_outputs, activation = None)
hidden3 = tf.layers.dense(encoder_outputs, n_hidden3, activation = tf.nn.relu)
hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.relu)
decoder_outputs = tf.layers.dense(hidden4, n_decoder_ouputs, activation = None)
    
recover_loss = tf.reduce_mean(tf.square(decoder_outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
train_op = optimizer.minimize(recover_loss)
    
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    n_epoches = 10
    batch_size = 150
    for epoch in range(n_epoches):
        for batch in range(mnist.train.num_examples//batch_size):
            X_batch, _ = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict = {X:X_batch})
    Z = sess.run(encoder_outputs, feed_dict = {X:mnist.train.images})
    plt.figure(0)
    plt.scatter(Z[:,0], Z[:,1], c = mnist.train.labels)
    plt.show()

















