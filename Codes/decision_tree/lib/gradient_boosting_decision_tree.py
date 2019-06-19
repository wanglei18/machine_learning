import numpy as np
from machine_learning.decision_tree.lib.decision_tree_regressor import DecisionTreeRegressor

class GBDT:        
    def __init__(self, num_trees, max_depth):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.trees = []
        
    def fit(self, X, y):
        r = y
        for t in range(self.num_trees):
            model = DecisionTreeRegressor(max_depth = self.max_depth)
            model.fit(X, r)
            self.trees.append(model)
            pred = model.predict(X)
            r = r - pred
    
    def predict(self,X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.sum(preds, axis=0)

    
        
        
               


























