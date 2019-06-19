import numpy as np

np.random.seed(0)

d = np.array([[2,0,0], [0,1,0], [0,0,0]])
p = np.random.rand(5,3)
q = np.random.rand(3,4)

x = p.dot(d).dot(q)

#print(x)
u,s,v = np.linalg.svd(x)
print(v)


a = np.array([[-0.45572083, -0.24473247],
 [-0.36452335, -0.78436792],
 [-0.46207643,  0.41263246],
 [-0.40806197,  0.38103889],
 [-0.52859442, 0.09704018]])
b = np.array([[4.72562351, 0], [0, 3.59300988e-01]])  
c = np.array([[-0.3547013,  -0.36251379, -0.67861873, -0.53126958]
 ,[ 0.52328055,  0.66650283, -0.27300628, -0.45543278]])

y = a.dot(b).dot(c)

z = x.T.dot(x)
#print(z)
eigvals, eigvecs = np.linalg.eig(z)
print(eigvecs.astype(float))


    
    






    
    
    










            
            
        
        
    



















