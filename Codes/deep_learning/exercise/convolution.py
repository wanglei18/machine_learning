import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape = [1, 4, 4, 1])
filters = np.zeros(shape=(2, 2, 1, 1), dtype=np.float32)
filters[0, 0, 0, 0] = 2
filters[0, 1, 0, 0] = -1
filters[1, 0, 0, 0] = -1
filters[1, 1, 0, 0] = 0
conv1 = tf.nn.conv2d(X, filters, strides = [1,1,1,1], padding = 'VALID')
conv2 = tf.nn.conv2d(X, filters, strides = [1,2,2,1], padding = 'VALID')
conv3 = tf.nn.conv2d(X, filters, strides = [1,1,1,1], padding = 'SAME')
pool1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1],
                                strides = [1,1,1,1], padding = 'VALID')
pool2 = tf.nn.avg_pool(conv3, ksize = [1,2,2,1],
                                strides = [1,1,1,1], padding = 'VALID')
    
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    V = np.array([[1,1,1,1],
              [1,10,10,1],
              [1,10,10,1],
              [1,1,1,1]]).reshape(1,4,4,1)
    output1 = sess.run(conv1, feed_dict={X: V})
    output2 = sess.run(conv2, feed_dict={X: V})
    output3 = sess.run(conv3, feed_dict={X: V})
    output4 = sess.run(pool1, feed_dict={X: V})
    output5 = sess.run(pool2, feed_dict={X: V})
    
    print("output1=", output1)
    print("output2=", output2)
    print("output3=", output3)
    print("output4=", output4)
    print("output5=", output5)









            
            
        
        
    



















