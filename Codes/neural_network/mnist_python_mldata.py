import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import machine_learning.neural_network.lib.neural_network as nn
from sklearn.metrics import accuracy_score

def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    return X

def convert_to_vector(y):
    m = len(y)
    k = np.max(y) + 1
    v = np.zeros(m * k).reshape(m,k)
    for i in range(m):
        v[i][y[i]] = 1
    return v

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

mnist = fetch_mldata('MNIST original', data_home="./")
m, n = mnist["data"].shape
X = mnist["data"]
y = mnist["target"].astype(np.int).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = process_features(X_train)
X_test = process_features(X_test)

layers = create_layers()
loss = nn.SoftmaxCrossEntropy()
model = nn.NeuralNetwork(layers, loss)
model.fit(X_train, convert_to_vector(y_train), 50000, 0.01)
v = model.predict(X_test)
proba = nn.softmax(v)
y_pred = np.argmax(proba, axis=1)
print(accuracy_score(y_test, y_pred))

"(55000, 10)"
"0.9749"





       
            
        
        
        
        
    
        
        
        
        
            
    

















