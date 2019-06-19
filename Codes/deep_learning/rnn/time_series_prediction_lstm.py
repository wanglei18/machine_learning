import tensorflow as tf
import numpy as np

def time_series(r):
    return r / 10.0 * np.sin(r / 10.0)

def get_samples(n_samples, r_max, R):
    r0 = np.random.rand(n_samples, 1) * (r_max - R)
    r = r0 + np.arange(0.0, R + 1) 
    f = time_series(r)
    x = f[:, 0:R].reshape(-1, R, 1)
    y = f[:, R].reshape(-1, 1)
    return x, y

R = 30
r_max = 100
np.random.seed(0)
X_train, y_train = get_samples(100, r_max, R)
X_test, y_test = get_samples(100, r_max, R)

n_inputs = 1
n_outputs = 1
num_units = 50
X = tf.placeholder(tf.float32, [None, R, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
states, final_state = tf.nn.dynamic_rnn(lstm_cell, X, dtype = tf.float32)
preds = tf.layers.dense(final_state.h, n_outputs)

loss = tf.reduce_mean(tf.square(preds - y)) 
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_op = optimizer.minimize(loss)

n_iterations = 1000
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for iteration in range(n_iterations):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_test, y: y_test})
            print(iteration, " MSE:", mse)
            

                
                
            
            








            
            
        
        
    



















