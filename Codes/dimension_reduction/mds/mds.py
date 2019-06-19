import numpy as np

class MDS:
    def __init__(self, n_components):
        self.d = n_components
        
    def get_distance_square(self,X):
        m,n = X.shape
        D = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                D[i][j] = np.linalg.norm(X[i]-X[j],2) ** 2
        return D
    
    def get_convariance(self, Dist):
        row_avg = np.average(Dist, axis = 0) 
        column_avg = np.average(Dist, axis = 1) 
        total_avg = np.average(Dist)
        m = len(Dist)
        B = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                B[i][j] = - (Dist[i][j] - column_avg[i] - row_avg[j] + total_avg) / 2.0
        return B
    
    def fit_transform(self, X):
        m, n = X.shape
        self.mean = X.mean(axis = 0)
        X = X - self.mean
        Dist = self.get_distance_square(X)
        B = self.get_convariance(Dist)
        eigen_values, eigen_vectors = np.linalg.eig(B) 
        pairs = [(eigen_values[i], eigen_vectors[:,i]) for i in range(m)]
        pairs = sorted(pairs, key = lambda pair: pair[0], reverse = True)
        VT = np.array([pairs[i][1] for i in range(self.d)])
        D = np.diag(np.array([pairs[i][0] for i in range(self.d)]))
        return np.sqrt(D).dot(VT).T


        
#np.random.seed(0)
#x = np.array([[0,1],[2,3]])
#model = MDS(n_components = 2)
#z = model.fit_transform(x)
#print(z)


#[[-0.42708022  1.34818489]
# [ 0.42708022 -1.34818489]]



    
    






    
    
    










            
            
        
        
    



















