import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(data_path, image_size, classes):
    images = []
    labels = []
    for c in classes:
        index = classes.index(c)
        path = os.path.join(data_path, c+'*.jpg')
        files = glob.glob(path)
        for file in files:
            image = cv2.imread(file)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1
            labels.append(label)
    print('finish loading')
    return np.array(images), np.array(labels)

classes = ['dog','cat']
n_classes = len(classes) 
img_size = 128
n_channels = 3
data_path = '/Users/wanglei/ML_Data/neural_network/cats_and_dogs_sampled4'
images, labels = load_data(data_path, img_size, classes)

#cv2.imshow('image0',images[0][:,:,0])
#cv2.imshow('image1',images[0][:,:,1])
#cv2.imshow('image2',images[0][:,:,2])
#cv2.imshow('image3',images[0])
#cv2.waitKey(0)

image_train, image_test, label_train, label_test = train_test_split(
    images, labels, test_size=0.1, random_state=0)

X = tf.placeholder(tf.float32, name="X",
                    shape = [None, img_size, img_size, n_channels])
y = tf.placeholder(tf.float32, name = "y", shape = [None, n_classes])
y_class = tf.argmax(y, axis = 1)

conv1 = tf.layers.conv2d(X, filters = 32, kernel_size = [3,3],
                        strides = [1,1], padding = 'SAME')
conv1_pool = tf.nn.max_pool(conv1, ksize = [1,2,2,1],
                        strides = [1,2,2,1], padding = 'SAME')
conv2 = tf.layers.conv2d(conv1_pool, filters = 32, kernel_size = [3, 3],
                        strides = [1,1], padding = 'SAME')
conv2_pool = tf.nn.max_pool(conv2, ksize = [1,2,2,1],
                        strides = [1,2,2,1], padding = 'SAME')
conv3 = tf.layers.conv2d(conv2_pool, filters = 64, kernel_size = [3,3],
                        strides=[1,1], padding = 'SAME')
conv3_pool = tf.nn.max_pool(conv3, ksize = [1,2,2,1],
                        strides = [1,2,2,1], padding = 'SAME')
n_features = conv3_pool.get_shape()[1:4].num_elements()
conv_flat = tf.reshape(conv3_pool, [-1, n_features])

fc1 = tf.layers.dense(conv_flat, 128, activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1, n_classes)
    
n_epoches = 20
batch_size = 32
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

y_pred = tf.nn.softmax(fc2, name='y_pred_proba')
y_class_pred = tf.argmax(y_pred, axis = 1)
correct_prediction = tf.equal(y_class_pred, y_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init=tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./cats_and_dogs_model/cats_and_dogs.ckpt")
    test_accuracy = accuracy.eval(feed_dict = {X:image_test, y:label_test})
    print("test accuracy=", test_accuracy)
    
    
    










            
            
        
        
    



















