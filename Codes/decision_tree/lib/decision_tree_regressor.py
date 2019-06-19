import numpy as np
from machine_learning.decision_tree.lib.decision_tree_base import DecisionTreeBase

def get_var(y, idx):
    y_avg = np.average(y[idx]) * np.ones(len(idx))
    return np.linalg.norm(y_avg - y[idx], 2) ** 2 / len(idx)

def get_r2(y_true, y_pred):
        numerator = (y_true - y_pred) **2
        denominator = (y_true - np.average(y_true)) ** 2
        return 1- numerator.sum() / denominator.sum()       

class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self, max_depth=0, feature_sample_rate=1.0):
        super().__init__(
            max_depth = max_depth, 
            feature_sample_rate = feature_sample_rate,
            get_score = get_var)
        
    
























