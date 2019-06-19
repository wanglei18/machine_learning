import numpy as np

class PCA:
    def __init__(self, n_components):
        self.d = n_components
        
    def fit_transform(self, X):
        self.mean = X.mean(axis = 0)
        X = X - self.mean
        U, D, VT = np.linalg.svd(X)
        self.W = VT[0:self.d].T
        return X.dot(self.W)
    
    def inverse_transform(self, Z):
        return Z.dot(self.W.T) + self.mean
        
np.random.seed(0)
x = np.random.rand(4,3)
pca = PCA(n_components = 2)
z = pca.fit_transform(x)
x_recovered = pca.inverse_transform(z)
print(x)
print(z)
print(x_recovered)

# [[ 0.5488135   0.71518937  0.60276338]
#  [ 0.54488318  0.4236548   0.64589411]
#  [ 0.43758721  0.891773    0.96366276]
#  [ 0.38344152  0.79172504  0.52889492]]
# [[-0.05781366  0.04867528]
#  [-0.2503891  -0.14842368]
#  [ 0.32293119 -0.09737747]
#  [-0.01472843  0.19712587]]
# [[ 0.47552238  0.68943431  0.6115416 ]
#  [ 0.56962374  0.43234882  0.64293089]
#  [ 0.44550231  0.89455443  0.96271475]
#  [ 0.42407699  0.80600465  0.52402793]]



    
    






    
    
    










            
            
        
        
    



















