import numpy as np
from machine_learning.support_vector_machine.lib.svm_smo import SVM

class KernelSVM(SVM):         
    def __init__(self, kernel = None):
        self.kernel = kernel
    
    def get_K(self, X_1, X_2):
        if self.kernel == None:
            return X_1.dot(X_2.T) 
        m1, m2 = len(X_1), len(X_2)
        K = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                K[i][j] = self.kernel(X_1[i], X_2[j])
        return K
        
    def fit(self, X, y, N=10):
        K = self.get_K(X, X) 
        self.smo(X, y, K, N)
        self.X_train = X
        self.y_train = y
         
    def predict(self, X):
        K = self.get_K(X, self.X_train)
        return np.sign(K.dot(self.Lambda * self.y_train) + self.b)





















