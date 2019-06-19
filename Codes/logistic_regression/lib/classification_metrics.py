import numpy as np

def cross_entropy(y_true, y_pred):
    return np.average(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def accuracy_score(y_true, y_pred):
    correct = (y_pred == y_true).astype(np.int)
    return np.average(correct)

def precision_score(y, z):
    tp = (z * y).sum()
    fp = (z * (1 - y)).sum()
    if tp + fp == 0:
        return 1.0
    else:
        return tp / (tp + fp)

def recall_score(y, z): 
    tp = (z * y).sum()
    fn = ((1 - z) * y).sum()
    if tp + fn == 0:
        return 1
    else:
        return tp / (tp + fn)
    

             





















