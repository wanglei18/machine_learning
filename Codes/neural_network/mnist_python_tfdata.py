import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import machine_learning.neural_network.lib.neural_network as nn
from sklearn.metrics import accuracy_score

def create_layers():
    n_features = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_classes = 10
    layers = []
    relu = nn.ReLUActivator()
    layers.append(nn.Layer(n_features, n_hidden1, activator = relu))
    layers.append(nn.Layer(n_hidden1, n_hidden2, activator = relu))
    layers.append(nn.Layer(n_hidden2, n_classes))
    return layers

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels
y_test_cls = np.argmax(y_test, axis=1)

layers = create_layers()
loss = nn.SoftmaxCrossEntropy()
model = nn.NeuralNetwork(layers, loss)
model.fit(X_train, y_train, 50000, 0.01)
v = model.predict(X_test)
proba = nn.softmax(v)
y_pred = np.argmax(proba, axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
print("accuracy = {}".format(accuracy))






       
            
        
        
        
        
    
        
        
        
        
            
    

















