from machine_learning.support_vector_machine.lib.svm_smo import SVM

class SoftSVM(SVM):
    def __init__(self, C = 1000):
        self.C = C
    
    def get_H(self, Lambda, i,j, y):
        C = self.C
        if y[i]==y[j]:
            return min(C, Lambda[i] + Lambda[j])
        else:
            return min(C, C + Lambda[j] - Lambda[i]) 
    
    def get_L(self, Lambda, i, j, y):
        if y[i]==y[j]:
            return max(0, Lambda[i] + Lambda[j] - self.C)
        else:
            return max(0, Lambda[j] - Lambda[i])
    
























