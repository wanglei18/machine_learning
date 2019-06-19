import tensorflow as tf
import numpy as np

n_features = 1
n_steps = 3
batch_size = 2
n_units = 5

X = tf.placeholder(tf.float32, [batch_size, n_steps, n_features])

lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = n_units)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
variables = lstm_cell.trainable_weights

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X_train = np.array([[[1],[2],[3]], [[10],[20],[30]]])
    
    outputs, final_state, variables = sess.run([outputs,states, variables], feed_dict={X: X_train})
    print(outputs.shape)
    #print(final_state.shape)
    print("outputs=", outputs)
    print("states=", final_state)
    print("variables=", variables)








            
            
        
        
    



















