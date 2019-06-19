import tensorflow as tf
import numpy as np

n_features = 1
n_steps = 3
batch_size = 2
n_units = [4, 5, 6]

X = tf.placeholder(tf.float32, [batch_size, n_steps, n_features])

layers = [tf.contrib.rnn.BasicRNNCell(num_units = n, activation = tf.nn.relu) for n in n_units]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
variables = multi_layer_cell.trainable_weights

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X_train = np.array([[[1],[2],[3]], [[10],[20],[30]]])
    
    outputs, final_state, variables = sess.run([outputs,states, variables], feed_dict={X: X_train})
    print("outputs=", outputs)
    print("final_state=", final_state)
    print("variables=", variables)








            
            
        
        
    



















