import numpy as np

def get_accuracy(y, z):
    I = (y == z).astype(np.int)
    return np.average(I)

def softmax(v):
    e = np.exp(v)
    s = e.sum(axis=0)
    for i in range(len(s)):
        e[i] /= s[i]
    return e

class IdentityActivator:
    def value(self, s):
        return s
    
    def derivative(self, s):
        return 1

class SigmoidActivator:
    def value(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def derivative(self, x):
        y = self.value(x)
        return y * (1 - y)
    
class ReLUActivator:
    def value(self, s):
        return np.maximum(0, s)
    
    def derivative(self, s):
        return (s > 0).astype(np.int)
    
class SoftmaxCrossEntropy:
    def value(self, y, v):
        p = softmax(v)
        return - (y * np.log(p)).sum()
    
    def derivative(self, y, v):
        p = softmax(v)
        return p - y
    
class MSE:
    def value(self, y, v):
        return (v - y) ** 2
    
    def derivative(self, y, v):
        return 2 * (v - y)

class Layer:
    def __init__(self, n_input, n_output, activator = IdentityActivator()):
        self.activator = activator
        r = np.sqrt(6.0 / (n_input + n_output))
        self.W = np.random.uniform(-r, r, (n_output, n_input))
        self.b = np.zeros((n_output, 1))
        self.outputs = np.zeros((n_output, 1))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.sums = self.W.dot(inputs) + self.b
        self.outputs = self.activator.value(self.sums)
    
    def back_propagation(self, delta_in, learning_rate):
        d = self.activator.derivative(self.sums) * delta_in
        self.delta_out = self.W.T.dot(d)
        self.W -= learning_rate * d.dot(self.inputs.T)
        self.b -= learning_rate * d
    
class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        
    def forward(self, x):
        layers = self.layers
        inputs = x
        for layer in layers:
            layer.forward(inputs)
            inputs = layer.outputs
        return inputs
    
    def back_propagation(self, y, outputs, learning_rate):
        delta_in = self.loss.derivative(y, outputs)
        for layer in self.layers[::-1]:
            layer.back_propagation(delta_in, learning_rate)
            delta_in = layer.delta_out
            
    def fit(self, X, y, N, learning_rate):
        for t in range(N):
            i = np.random.randint(0, len(X))
            outputs = self.forward(X[i].reshape(-1,1))             
            self.back_propagation(y[i].reshape(-1,1), outputs, learning_rate)
              
    def predict(self, X):
        y = []
        for i in range(len(X)):
            p = self.forward(X[i].reshape(-1,1)).reshape(-1)
            y.append(p)
        return np.array(y)    
    








       
            
        
        
        
        
    
        
        
        
        
            
    

















