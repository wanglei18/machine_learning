import numpy as np
from machine_learning.decision_tree.lib.decision_tree_base import DecisionTreeBase

def get_impurity(y, idx):
    p = np.average(y[idx], axis=0)  
    return 1 - p.dot(p.T)

def get_entropy(y, idx):
    _, k = y.shape
    p = np.average(y[idx], axis=0) 
    return - np.log(p + 0.001 * np.random.rand(k)).dot(p.T)
    
class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, max_depth=0, feature_sample_rate=1.0):
        super().__init__(max_depth = max_depth, 
            feature_sample_rate = feature_sample_rate,
            get_score = get_entropy)
    
    def predict_proba(self, X):
        return super().predict(X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba,axis=1)




























