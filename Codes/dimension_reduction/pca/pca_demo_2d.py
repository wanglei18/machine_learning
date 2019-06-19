import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)        

def generate_samples(m):
    r = 2 * np.random.rand(m,1) - np.ones((m,1))
    return r.dot(np.ones((1,2))) + np.random.normal(0, 0.05,(m,2))

m = 50
x = generate_samples(m)
plt.figure(1)
plt.axis([-2,2,-2,2])
plt.scatter(x[:,0], x[:,1], s=10)


w = np.array([1,0]).reshape(2,1)
z = x.dot(w)
plt.figure(2)
plt.axis([-2,2,-2,2])
plt.scatter(z[:], 1.5*np.ones(m), s=10, c='r')

w = np.array([1/np.sqrt(2),1/np.sqrt(2)]).reshape(2,1)
z = x.dot(w)
plt.scatter(z[:], 0 * np.ones(m), s=10, c='g')

w = np.array([1/np.sqrt(2),-1/np.sqrt(2)]).reshape(2,1)
z = x.dot(w)
plt.scatter(z[:], -1.5 * np.ones(m), s=10, c='m')


plt.show()



    
    






    
    
    










            
            
        
        
    



















