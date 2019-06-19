import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape = [1, 3, 3, 1])
conv1 = tf.layers.conv2d(X, filters=2, kernel_size=2, padding = 'SAME')
pool1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1],
                                strides = [1,1,1,1], padding = 'VALID')
conv2 = tf.layers.conv2d(pool1, filters=2, kernel_size=[1,2], padding = 'VALID')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    V = np.array([[6,9,8],
              [1,3,7],
              [2,4,5]]).reshape(1,3,3,1)
    output1 = sess.run(conv1, feed_dict={X: V})
    output2 = sess.run(pool1, feed_dict={X: V})
    output3 = sess.run(conv2, feed_dict={X: V})
    
    print("output1=", output1.shape)
    print("output2=", output2.shape)
    print("output3=", output3.shape)








            
            
        
        
    



















