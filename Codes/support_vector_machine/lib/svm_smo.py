import numpy as np

class SVM:
    def get_H(self, Lambda, i, j, y):
        if y[i]==y[j]:
            return Lambda[i] + Lambda[j]
        else:
            return float("inf")
    
    def get_L(self, Lambda, i, j, y):
        if y[i]==y[j]:
            return 0.0
        else:
            return max(0, Lambda[j] - Lambda[i])
            
    def smo(self, X, y, K, N):
        m, n = X.shape
        Lambda = np.zeros((m,1))
        epsilon = 1e-6
        for t in range(N):
            for i in range(m):
                for j in range(m):
                    D_ij = 2 * K[i][j] - K[i][i] - K[j][j]
                    if abs(D_ij) < epsilon:
                        continue
                    E_i = K[:, i].dot(Lambda * y) - y[i]
                    E_j = K[:, j].dot(Lambda * y) - y[j]
                    delta_j = 1.0 * y[j] * (E_j - E_i) / D_ij
                    H_ij = self.get_H(Lambda, i, j, y)
                    L_ij = self.get_L(Lambda, i, j, y)
                    if Lambda[j] + delta_j > H_ij:
                        delta_j = H_ij - Lambda[j]
                        Lambda[j] = H_ij
                    elif Lambda[j] + delta_j < L_ij:
                        delta_j = L_ij - Lambda[j]
                        Lambda[j] = L_ij
                    else:
                        Lambda[j] += delta_j
                    delta_i = - y[i] * y[j] * delta_j
                    Lambda[i] += delta_i                 
                    if Lambda[i] > epsilon:
                        b = y[i] - K[:, i].dot(Lambda * y)
                    elif Lambda[j] > epsilon:
                        b = y[j] - K[:, j].dot(Lambda * y)
        self.Lambda = Lambda
        self.b = b
        
    def fit(self, X, y, N = 10):
        K = X.dot(X.T)
        self.smo(X, y, K, N)
        self.w = X.T.dot(self.Lambda * y)
        
    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)























