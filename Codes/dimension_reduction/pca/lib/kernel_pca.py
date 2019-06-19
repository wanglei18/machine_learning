import numpy as np

def default_kernel(x1, x2):
    return x1.dot(x2.T)

class KernelPCA:    
    def __init__(self, n_components, kernel = None):
        self.d = n_components
        self.kernel = default_kernel
        if kernel != None:
            self.kernel = kernel
        
    def fit_transform(self, X):
        m,n = X.shape
        K = np.zeros((m,m))
        for s in range(m):
            for r in range(m):
                K[s][r] = self.kernel(X[s],X[r])
        J = np.ones((m,m)) * (1.0 / m)
        K = K - J.dot(K) -K.dot(J) + J.dot(K).dot(J)
        eigen_values, eigen_vectors = np.linalg.eig(K) 
        pairs = [(eigen_values[i], eigen_vectors[:,i]) for i in range(m)]
        pairs = sorted(pairs, key = lambda pair: pair[0], reverse = True)
        Z = np.array([pairs[i][1] * np.sqrt(pairs[i][0]) for i in range(self.d)]).T
        return Z  
    

    

        
        
        
        
        






    
    
    










            
            
        
        
    



















