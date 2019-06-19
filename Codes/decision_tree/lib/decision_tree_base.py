import numpy as np

class Node:
    j = None
    theta = None
    p = None
    left = None
    right = None

class DecisionTreeBase:    
    def __init__(self, max_depth, feature_sample_rate, get_score):
        self.max_depth = max_depth
        self.feature_sample_rate = feature_sample_rate
        self.get_score = get_score
        
    def split_data(self, j, theta, X, idx):
        idx1, idx2 = list(), list()
        for i in idx:
            value = X[i][j]
            if value <= theta:
                idx1.append(i)
            else:
                idx2.append(i)
        return idx1, idx2
    
    def get_random_features(self, n):
        shuffled = np.random.permutation(n)
        size = int(self.feature_sample_rate * n)
        selected = shuffled[:size]
        return selected
    
    def find_best_split(self, X, y, idx):
        m, n = X.shape
        best_score = float("inf")
        best_j = -1
        best_theta = float("inf")
        best_idx1, best_idx2 = list(), list()
        selected_j = self.get_random_features(n)
        for j in selected_j:
            thetas = set([x[j] for x in X])
            for theta in thetas:
                idx1, idx2 = self.split_data(j, theta, X, idx)
                if min(len(idx1), len(idx2)) == 0 :
                    continue
                score1, score2 = self.get_score(y, idx1), self.get_score(y, idx2)
                w = 1.0 * len(idx1) / len(idx)
                score = w * score1 + (1-w) * score2 
                if score < best_score:
                    best_score = score
                    best_j = j
                    best_theta = theta
                    best_idx1 = idx1
                    best_idx2 = idx2
        return best_j, best_theta, best_idx1, best_idx2, best_score
             
    def generate_tree(self, X, y, idx, d):
        r = Node()
        r.p = np.average(y[idx], axis=0)
        if d == 0 or len(idx)<2:
            return r
        current_score = self.get_score(y, idx)
        j, theta, idx1, idx2, score = self.find_best_split(X, y, idx)
        if score >= current_score:
            return r
        r.j = j
        r.theta = theta
        r.left = self.generate_tree(X, y, idx1, d-1)
        r.right = self.generate_tree(X, y, idx2, d-1)
        return r
        
    def fit(self, X, y):
        self.root = self.generate_tree(X, y, range(len(X)), self.max_depth) 
        
    def get_prediction(self, r, x):
        if r.left == None and r.right == None:
            return r.p
        value = x[r.j]
        if value <= r.theta:
            return self.get_prediction(r.left, x)
        else:
            return self.get_prediction(r.right, x)
            
    def predict(self, X):
        y = list()
        for i in range(len(X)):
            y.append(self.get_prediction(self.root, X[i]))
        return np.array(y)
    





















