import numpy as np

class MDS:
    def __init__(self, n_components):
        self.d = n_components
        
    def fit_transform(self, X):
        m, n = X.shape
        self.mean = X.mean(axis = 0)
        X = X - self.mean
        B = X.dot(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(B) 
        pairs = [(eigen_values[i], eigen_vectors[:,i]) for i in range(m)]
        pairs = sorted(pairs, key = lambda pair: pair[0], reverse = True)
        Z = np.array([np.sqrt(pairs[i][0]) * pairs[i][1] for i in range(self.d)]).T
        return Z


        
#np.random.seed(0)
#x = np.array([[0,1],[2,3]])
#model = MDS(n_components = 2)
#z = model.fit_transform(x)
#print(z)


#[[-0.42708022  1.34818489]
# [ 0.42708022 -1.34818489]]



    
    






    
    
    










            
            
        
        
    



















