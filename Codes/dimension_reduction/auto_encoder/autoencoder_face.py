import numpy as np
import tensorflow as tf
import cv2
from skimage import io
import os
import matplotlib.pyplot as plt

def read_face_images(data_folder):
    image_paths = [os.path.join(data_folder, item) for item in os.listdir(data_folder)]
    images = []
    labels = []
    subjects = []
    for image_path in image_paths:
        im = io.imread(image_path,as_grey=True)
        images.append(np.array(im, dtype='uint8'))
        labels.append(int(os.path.split(image_path)[1].split(".")[0].replace("subject", "")))
        subjects.append(os.path.split(image_path)[1].split(".")[1])
    return np.array(images), np.array(labels), np.array(subjects)

data_folder = "/Users/wanglei/ml_data/yalefaces"
images, labels, subjects = read_face_images(data_folder)

#plt.imshow(images[1])
#plt.show()

m, n1, n2 = images.shape
X_train = images.reshape(m,-1)

n_features = n1 * n2
n_hidden1 = 300
n_hidden2 = 150
n_encoder_outputs = 10
n_hidden3 = n_hidden2
n_hidden4 = n_hidden1
n_decoder_ouputs = n_features 
X = tf.placeholder(tf.float32, shape=(None, n_features))
hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu)
encoder_outputs = tf.layers.dense(hidden2, n_encoder_outputs, activation = tf.nn.relu)
hidden3 = tf.layers.dense(encoder_outputs, n_hidden3, activation = tf.nn.relu)
hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.relu)
decoder_outputs = tf.layers.dense(hidden4, n_decoder_ouputs, activation = None)
    
reconstruct_loss = tf.reduce_mean(tf.square(decoder_outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
train_op = optimizer.minimize(reconstruct_loss)

saver = tf.train.Saver()    
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_epoches = 1000
    for epoch in range(n_epoches):
        sess.run(train_op, feed_dict = {X: X_train})
        #_, loss = sess.run([train_op, reconstruct_loss], feed_dict = {X: X_train})
        #print("epoch ", epoch, ": loss=", loss)
    saver.save(sess, "./autoencoder_face_model/autoencoder_face.ckpt")
    



    



















