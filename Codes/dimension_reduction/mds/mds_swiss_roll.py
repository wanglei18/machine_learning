import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import PythonCodes.dimension_reduction.mds.mds2 as mds

np.random.seed(0)
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)
model = mds.MDS(n_components=2)
Z = model.fit_transform(X)

plt.figure(20)
plt.scatter(Z[:, 0], Z[:, 1], c=color)
plt.show()


        
#np.random.seed(0)
#x = np.array([[0,1],[2,3]])
#model = MDS(n_components = 2)
#z = model.fit_transform(x)
#print(z)


#[[-0.42708022  1.34818489]
# [ 0.42708022 -1.34818489]]



    
    






    
    
    










            
            
        
        
    



















