import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape = [1, 3, 3, 1])
filters1 = np.zeros(shape=(2, 2, 1, 2), dtype=np.float32)
filters1[0, 0, 0, 0] = 1
filters1[0, 1, 0, 0] = 0
filters1[1, 0, 0, 0] = 0
filters1[1, 1, 0, 0] = -1
filters1[0, 0, 0, 1] = 0
filters1[0, 1, 0, 1] = -1
filters1[1, 0, 0, 1] = 1
filters1[1, 1, 0, 1] = 0

conv1 = tf.nn.conv2d(X, filters1, strides = [1,1,1,1], padding = 'SAME')
pool1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1],
                                strides = [1,1,1,1], padding = 'VALID')

filters2 = np.zeros(shape=(1, 2, 2, 2), dtype=np.float32)
filters2[0, 0, 0, 0] = 1
filters2[0, 1, 0, 0] = 0
filters2[0, 0, 0, 1] = 0
filters2[0, 1, 0, 1] = 1
filters2[0, 0, 1, 0] = 1
filters2[0, 1, 1, 0] = 0
filters2[0, 0, 1, 1] = 0
filters2[0, 1, 1, 1] = 1
conv2 = tf.nn.conv2d(pool1, filters2, strides = [1,1,1,1], padding = 'VALID')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    V = np.array([[6,9,8],
              [1,3,7],
              [2,4,5]]).reshape(1,3,3,1)
    output1 = sess.run(conv1, feed_dict={X: V})
    output2 = sess.run(pool1, feed_dict={X: V})
    output3 = sess.run(conv2, feed_dict={X: V})
    
    print("output1=", output1)
    print("output2=", output2)
    print("output3=", output3)








            
            
        
        
    



















