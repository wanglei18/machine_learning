import numpy as np
from sklearn.neighbors import NearestNeighbors

class LLE:
    def __init__(self, n_components, n_neighbors):
        self.d = n_components
        self.k = n_neighbors
    
    def get_weights(self, X, knn):
        m, n = X.shape
        W = np.zeros((m,m))
        for i in range(m):
            U = X[knn[i]].reshape(-1,n)
            k = len(U)
            for t in range(k):
                U[t] -= X[i]
            C = U.dot(U.T)
            w = np.linalg.inv(C).dot(np.ones((k,1)))
            w /= w.sum(axis=0)
            for t in range(k):
                W[i][knn[i][t]] = w[t]
        return W
        
    def fit_transform(self,X):
        m, n = X.shape
        model = NearestNeighbors(n_neighbors=self.k+1).fit(X)
        knn = model.kneighbors(X, return_distance = False)[:,1:]
        W = self.get_weights(X, knn)
        M = (np.identity(m) - W).T.dot(np.identity(m) - W)
        eigen_values, eigen_vectors = np.linalg.eig(M) 
        pairs = [(eigen_values[i], eigen_vectors[:,i]) for i in range(m)]
        pairs = sorted(pairs, key = lambda pair: pair[0])
        Z = np.array([pairs[i+1][1] for i in range(self.d)]).T
        return Z

np.random.seed(0)
x = np.random.rand(4,3)
lle = LLE(n_components = 2, n_neighbors=2)
z = lle.fit_transform(x)
print(z)


#[[ 0.01174409  0.20821725]
# [-0.77112287  0.20410023]
# [ 0.1379214  -0.85035817]
# [ 0.62145738  0.43804069]]
















