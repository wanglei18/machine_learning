import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from machine_learning.decision_tree.lib.decision_tree_regressor import DecisionTreeRegressor
from machine_learning.decision_tree.lib.random_forest_regressor import RandomForestRegressor
from sklearn.metrics import r2_score

def get_data():
    df = pd.read_csv("/Users/wanglei/ML_data/decision_tree/bike.csv")
    df.datetime = df.datetime.apply(pd.to_datetime)
    df['hour'] = df.datetime.apply(lambda x: x.hour)
    y = df['count'].values
    df.drop(['datetime','casual','registered','count'], 1, inplace = True)
    X = df.values
    return X, y

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=3)

model = DecisionTreeRegressor(max_depth = 8)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("tree r2 = {}".format(r2_score(y_test, y_pred)))

model = RandomForestRegressor(
    max_depth = 8, 
    num_trees=100, 
    feature_sample_rate=1.0, 
    data_sample_rate=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("forest r2= {}".format(r2_score(y_test, y_pred)))

#tree r2= 0.772880220951
#forest r2= 0.815791610688
























