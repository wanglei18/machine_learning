import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def time_series(r):
    return r / 10.0 * np.sin(r / 10.0)

def get_samples(n_samples, r_max, R):
    r0 = np.random.rand(n_samples, 1) * (r_max - R)
    r = r0 + np.arange(0.0, R + 1) 
    f = time_series(r)
    x = f[:, 0:R].reshape(-1, R, 1)
    y = f[:, R].reshape(-1, 1)
    return x, y

n_inputs = 1
n_outputs = 1
X = tf.placeholder(tf.float32, [None, R, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units = 50, activation = tf.nn.relu)
states, final_state = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
preds = tf.layers.dense(final_state, n_outputs)

loss = tf.reduce_mean(tf.square(preds - y)) 
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_op = optimizer.minimize(loss)

n_iterations = 1000
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    R = 30
    r_max = 100
    np.random.seed(0)
    X_train, y_train = get_samples(100, r_max, R)
    X_test, y_test = get_samples(100, r_max, R)
    for iteration in range(n_iterations):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_test, y: y_test})
            print(iteration, " MSE:", mse)
            
    
    true_vals = []
    pred_vals = []
    for i in range(50):
        r = np.linspace(i, i + R, R)
        x = time_series(r).reshape(-1, R, 1)
        pred_val = sess.run(preds, feed_dict={X: x}).reshape(-1)[0]
        true_val = time_series(i+R)
        pred_vals.append(pred_val)
        true_vals.append(true_val)
    plt.figure(27)
    plt.plot(true_vals, c="r")
    plt.plot(pred_vals, c="b")
    #plt.axis([0, 100, -10, 10])
    plt.show()
              
                
            
            








            
            
        
        
    



















