import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("decision tree r2=", r2_score(y_test, y_pred))

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("linear regression r2=", r2_score(y_test, y_pred))








#plt.show()























