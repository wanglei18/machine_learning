import numpy as np
import matplotlib.pyplot as plt

def threshold(t, proba):
    return (proba >= t).astype(np.int)

def plot_roc_curve(proba, y):
    fpr, tpr= [], []
    for i in range(100):
        z = threshold(0.01 * i, proba)
        tp = (y * z).sum()
        fp = ((1 - y) * z).sum()
        tn = ((1 - y) * (1 - z)).sum()
        fn = (y * (1 - z)).sum()
        fpr.append(1.0 * fp / (fp + tn))
        tpr.append(1.0 * tp / (tp + fn))  
    plt.plot(fpr, tpr)
    plt.show()
        




















