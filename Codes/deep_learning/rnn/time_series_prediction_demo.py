import numpy as np
import matplotlib.pyplot as plt

def time_series(r):
    return r / 10.0 * np.sin(r / 10.0) 

n_steps = 30
step_size = 1
r_min = 0
r_max = 100
np.random.seed(0)

r = np.linspace(r_min, r_max, int((r_max - r_min) / step_size))

plt.figure(0)
r_instance = np.linspace(20, 20 + step_size * (n_steps + 1), n_steps + 1)
plt.plot(r, time_series(r), c="y")
plt.plot(r_instance[:-1], time_series(r_instance[:-1]), "b-", linewidth=3)
plt.axis([0, 100, -10, 10])

plt.figure(1)
r_instance = np.linspace(50, 50 + step_size * (n_steps + 1), n_steps + 1)
plt.plot(r, time_series(r), c="y")
plt.plot(r_instance[:-1], time_series(r_instance[:-1]), "b-", linewidth=3)
plt.axis([0, 100, -10, 10])

plt.figure(3)
#plt.axis([4, 9, -6, 10])
plt.plot(r_instance[:30], time_series(r_instance[:30]), "bo", markersize=5, label="instance")
plt.show()









            
            
        
        
    



















