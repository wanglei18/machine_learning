import numpy as np

class LDA:
    def __init__(self, n_components):
        self.d = n_components
        
    def fit_transform(self, X, y):
        sums = dict()
        counts = dict()
        m,n = X.shape  
        for t in range(m):
            i = y[t]
            if i not in sums:
                sums[i] = np.zeros((1,n))
                counts[i] = 0
            sums[i] += X[t].reshape(1,n)
            counts[i] += 1
        X_mean = np.mean(X, axis=0).reshape(1,n)
        S_b = np.zeros((n,n))
        for i in counts:
            v = X_mean - 1.0 * sums[i] / counts[i] 
            S_b += counts[i] * v.T.dot(v)
        S_w = np.zeros((n,n))
        for t in range(m):
            i = y[t]
            u = X[t].reshape(1,n) - 1.0 * sums[i] / counts[i]
            S_w += u.T.dot(u)
        A = np.linalg.pinv(S_w).dot(S_b)
        values, vectors = np.linalg.eig(A) 
        pairs = [(values[j], vectors[:,j]) for j in range(len(values))]
        pairs = sorted(pairs, key = lambda pair: pair[0], reverse = True)
        W = np.array([pairs[j][1] for j in range(self.d)]).T
        return X.dot(W) 
           




    
    






    
    
    










            
            
        
        
    



















