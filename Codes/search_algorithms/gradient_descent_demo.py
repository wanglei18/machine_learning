'''
import numpy as np
import matplotlib.pyplot as plt

def F(w):
    return (w - 1) ** 2

X, y = [], []
eta, epsilon = 0.1, 0.01
w = 0
while abs(2 * (w - 1)) > epsilon:
    X.append(w)
    y.append(F(w))
    w = w - eta * 2 * (w - 1)

W = np.linspace(0, 2, 100).reshape(100, 1)
U = F(W)
plt.plot(W, U)
plt.scatter(X, y, s=15)
plt.show()
'''
eta, epsilon = 0.1, 0.01
w = 0
while abs(2 * (w - 1)) > epsilon:
    w = w - eta * 2 * (w - 1)
print(w)
    
    
    









