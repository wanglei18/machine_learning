import numpy as np
from machine_learning.linear_regression.lib.stepwise_regression import StepwiseRegression
from machine_learning.linear_regression.lib.stagewise_regression import StagewiseRegression

X= np.array(
[[ 0.06,  0.34,  0.03]
 ,[ 0.44,  0.76,  0.28]
 ,[ 0.86,  0.44,  0.20]
 ,[ 0.26,  0.09,   0.25]])
y = np.array(
[[ 0.42]
 ,[ 1.32 ]
 ,[ 0.84]
 ,[ 0.61]])

model = StepwiseRegression()
model.forward_selection(X, y)
print(model.w)

model = StagewiseRegression()
model.feature_selection(X, y, N=1000, eta=0.01)
print(model.w)






     













