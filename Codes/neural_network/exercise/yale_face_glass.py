import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from skimage import io
import os

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

def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    return X

n_features = 243 * 320
n_hidden1 = 500
n_hidden2 = 500
n_classes = 2
X = tf.placeholder(tf.float32, shape=(None, n_features))
y = tf.placeholder(tf.int64, shape=(None))  
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
outputs = tf.layers.dense(hidden2, n_classes)  
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=outputs)
loss = tf.reduce_mean(cross_entropy)
learning_rate = 0.01
n_epoches = 100
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(outputs, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    data_folder = "/Users/wanglei/ml_data/yalefaces"
    images, labels, subjects = read_face_images(data_folder)
    m, n1, n2 = images.shape
    images = images.reshape(m,-1)
    subjects = np.array([subjects=='glasses']).astype(int).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(
        images, subjects, test_size=0.2, random_state=0)
    X_train = process_features(X_train)
    X_train = process_features(X_test)
    for epoch in range(n_epoches):
        idx = np.random.permutation(len(X_train))
        for i in idx:
            X_i = X_train[i].reshape(1,-1)
            y_i = y_train[i].reshape(-1)
            sess.run(train_op, feed_dict = {X : X_i, y : y_i})
        test_acc = accuracy.eval(feed_dict = {X : X_test, y : y_test})
        print("epoch ", epoch, ": test accuracy=", test_acc)



#print(images.shape)
#print(labels.shape)
#print(subjects.shape)
#print(set(subjects))
#plt.imshow(images[1])
#plt.show()








    



















