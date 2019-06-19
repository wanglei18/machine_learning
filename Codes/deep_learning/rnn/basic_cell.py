import tensorflow as tf
import numpy as np

n_features = 1
n_steps = 3
batch_size = 2
n_units = 4

X = tf.placeholder(tf.float32, [batch_size, n_steps, n_features])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_units, activation = tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
variables = basic_cell.trainable_weights

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X_train = np.array([[[1],[2],[3]], [[10],[20],[30]]])
    
    outputs, final_state, variables = sess.run([outputs,states, variables], feed_dict={X: X_train})
    #print(outputs.shape)
    #print(final_state.shape)
    print("outputs=", outputs)
    #print("states=", final_state)
    print("variables=", variables)

# outputs= [[[ 0.60012317  0.62252373  0.57632923  0.53898317]
#  [ 0.7781719   0.7656675   0.64542878  0.67971325]
#  [ 0.8563956   0.84386891  0.72526568  0.7444104 ]]

# [[ 0.9830398   0.99332535  0.95594341  0.8267113 ]
#  [ 0.99985301  0.99996555  0.99778783  0.97855753]
#  [ 0.9999975   0.99999976  0.99990129  0.99565375]]]
#variables= [array([[ 0.40597832,  0.50027394,  0.30772233,  0.15624964],
#       [ 0.33880889, -0.75434303,  0.62405837,  0.74295783],
#       [ 0.28548503,  0.56230855, -0.38035628,  0.02667302],
#       [-0.08919567,  0.2195152 , -0.58481157, -0.36977959],
#       [ 0.21048081,  0.29610264,  0.3392942 ,  0.35362744]], dtype=float32), array([ 0.,  0.,  0.,  0.], dtype=float32)]







            
            
        
        
    



















