import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import machine_learning.neural_network.lib.neural_network as nn
from sklearn.metrics import r2_score

def process_features(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(X)  
    return X

def create_layers():
    n_features = 8
    n_hidden1 = 100
    n_hidden2 = 50
    n_outputs = 1
    layers = []
    relu = nn.ReLUActivator()
    layers.append(nn.Layer(n_features, n_hidden1, activator = relu))
    layers.append(nn.Layer(n_hidden1, n_hidden2, activator = relu))
    layers.append(nn.Layer(n_hidden2, n_outputs))
    return layers
    
housing = fetch_california_housing()
X = housing.data  
y = housing.target.reshape(-1,1)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_train = process_features(X_train)
X_test = process_features(X_test)

layers = create_layers()
loss = nn.MSE()
model = nn.NeuralNetwork(layers, loss)
model.fit(X_train, y_train, 100000, 0.01)
y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))








       
            
        
        
        
        
    
        
        
        
        
            
    

















