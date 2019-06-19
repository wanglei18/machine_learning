import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from machine_learning.dimension_reduction.lda.lib.lda import LDA

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
X, Y = mnist.train.images, mnist.train.labels
model = LDA(n_components = 2)
Z = model.fit_transform(X, Y)
plt.figure(6)
plt.scatter(Z[:,0], Z[:,1], c = Y)
plt.show()








    
    






    
    
    










            
            
        
        
    



















