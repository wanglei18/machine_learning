import numpy as np
import matplotlib.pyplot as plt

def f(w):
    return w ** 2

x, y = [], []
eta, epsilon = 0.1, 0.01
w = -1.5
while abs(f(w)) > epsilon:
    x.append(w)
    y.append(f(w))
    w = w - f(w) / (2 * w)

W = np.linspace(-2, 2, 100).reshape(100, 1)
u = f(W)
plt.plot(W, u)
plt.scatter(x, y)
plt.show()

'''
eta, epsilon = 0.1, 0.01
w = 0
while abs(2 * (w - 1)) > epsilon:
    w = w - eta * 2 * (w - 1)
print(w)
'''
    
    









