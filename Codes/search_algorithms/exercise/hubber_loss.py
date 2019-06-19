import numpy as np
import matplotlib.pyplot as plt

def hubber(r, epsilon):
    if abs(r) < epsilon:
        return r ** 2
    else:
        return 2 * epsilon * abs(r) - epsilon ** 2
        
X = np.linspace(-2, 2, 100).reshape(100, 1)
Y = np.array([hubber(x, 0.5) for x in X])
plt.axis([-2,2,0,2])
plt.plot(X, Y)
plt.show()


















