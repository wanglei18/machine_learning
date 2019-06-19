import numpy as np
from sklearn.manifold import LocallyLinearEmbedding



np.random.seed(0)
x = np.random.rand(4,3)
lle = LocallyLinearEmbedding(n_components = 2, n_neighbors=2)
z = lle.fit_transform(x)
print(z)

#[[ 0.01174409  0.20821725]
# [-0.77112287  0.20410023]
# [ 0.1379214  -0.85035817]
# [ 0.62145738  0.43804069]]

















