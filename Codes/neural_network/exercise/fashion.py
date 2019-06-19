import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def get_data():
    df = pd.read_csv("/Users/wanglei/ML_data/fashion/fashion-mnist_train.csv")
    y = df['label'].values
    df.drop(['label'], 1, inplace = True)
    X = df.values
    return X, y

def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    return X

n_features = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_classes = 10
X = tf.placeholder(tf.float32, shape=(None, n_features))
y = tf.placeholder(tf.int64, shape=(None))
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
outputs = tf.layers.dense(hidden2, n_classes)  
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=outputs)
loss = tf.reduce_mean(cross_entropy)  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)  
correct = tf.nn.in_top_k(outputs, y, 1)
accuracy_score = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    images,labels = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=0)
    X_train = process_features(X_train)
    X_test = process_features(X_test)
    for t in range(50000):
        i = np.random.randint(0, len(X_train))
        X_i = X_train[i].reshape(1,-1)
        y_i = y_train[i].reshape(-1)
        sess.run(train_op, feed_dict = {X : X_i, y : y_i})
    accuracy = accuracy_score.eval(feed_dict = {X : X_test, y : y_test})
    print("accuracy = {}".format(accuracy))


























