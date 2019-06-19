import numpy as np
from sklearn.manifold import MDS
       
np.random.seed(0)
#x = np.random.rand(4,3)
x = np.array([[0,1],[2,3]])
model = MDS(n_components = 2, metric=False)
z = model.fit_transform(x)
print(z)

print(model.get_params())


#[[ 0.06135802 -0.01593326]
# [-0.00745756 -0.2913253 ]
# [-0.25541641  0.2224335 ]
# [ 0.20151595  0.08482507]]



    
    






    
    
    










            
            
        
        
    



















