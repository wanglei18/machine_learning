import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mnist = fetch_mldata('MNIST original', data_home="./")
X = mnist["data"]
Y = mnist["target"]

model = LinearDiscriminantAnalysis(n_components = 2)
Z = model.fit_transform(X, Y)

plt.figure(6)
plt.scatter(Z[:,0], Z[:,1], c = Y)

plt.show()








    
    






    
    
    










            
            
        
        
    



















