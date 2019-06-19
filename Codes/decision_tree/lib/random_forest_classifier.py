import numpy as np
from machine_learning.decision_tree.lib.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:        
    def __init__(self, num_trees,max_depth, feature_sample_rate, 
            data_sample_rate, random_state = 0):
        self.max_depth, self.num_trees = max_depth, num_trees
        self.feature_sample_rate = feature_sample_rate
        self.data_sample_rate = data_sample_rate
        self.trees = []
        np.random.seed(random_state)
    
    def get_data_samples(self, X, y):
        shuffled_indices = np.random.permutation(len(X))
        size = int(self.data_sample_rate * len(X))
        selected_indices = shuffled_indices[:size]
        return X[selected_indices], y[selected_indices]
    
    def fit(self, X, y):
        for t in range(self.num_trees):
            X_t, y_t = self.get_data_samples(X, y)
            model = DecisionTreeClassifier(
                max_depth = self.max_depth, 
                feature_sample_rate = self.feature_sample_rate)
            model.fit(X_t, y_t)
            self.trees.append(model)
    
    def predict_proba(self, X):
        probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.average(probas, axis=0)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)   
    
    
               


























