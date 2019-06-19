import numpy as np
from scipy.stats import f

class StepwiseRegression:       
    def fit(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
    def compute_mse(self, X, y):
        w = self.fit(X,y)
        r = y - X.dot(w)
        return r.T.dot(r)
    
    def f_test(self, mse_selected, mse_min, m):
        if mse_min > mse_selected:
            return False
        F = mse_selected / mse_min
        p_value = f.cdf(F, m, m)
        return p_value > 0.95
        
    def forward_selection(self, X, y):
        m,n = X.shape 
        A, C = [0], [i for i in range(1,n)]
        for i in range(n-1):
            MSE_A = self.compute_mse(X[:,A], y)
            MSE_min, j_min = float("inf"), -1
            j_min = -1
            for j in C:
                MSE_j = self.compute_mse(X[:, A+[j]], y)
                if MSE_j < MSE_min:
                    MSE_min, j_min = MSE_j, j
            if self.f_test(MSE_A, MSE_min, m):
                A.append(j_min)
                C.remove(j_min)
            else:
                break
        self.w = self.fit(X[:, A], y)
        self.A = A
    
    def predict(self, X):
        return X[:, self.A].dot(self.w)





     













