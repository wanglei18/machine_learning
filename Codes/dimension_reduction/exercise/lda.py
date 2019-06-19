import numpy as np
from machine_learning.dimension_reduction.lda.lib.lda import LDA

X = np.array([[1,1,0], [0,1,1], [1,0,1],[0,1,0]])
y = np.array([0,0,1,1]) 
model = LDA(n_components=2)
Z = model.fit_transform(X, y)
print(Z)         




    
    






    
    
    










            
            
        
        
    



















