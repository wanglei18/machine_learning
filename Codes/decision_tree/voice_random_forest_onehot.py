import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from machine_learning.decision_tree.lib.decision_tree_classifier import DecisionTreeClassifier
from machine_learning.decision_tree.lib.random_forest_classifier import RandomForestClassifier
from sklearn.metrics import accuracy_score
   
def get_data():
    df = pd.read_csv("/Users/wanglei/ML_data/decision_tree/voice.csv")
    y = (df["label"].values=='male').astype(np.int)
    df.drop(['label'], 1, inplace = True)
    X = df.values
    return X, y.reshape(-1,1)

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train).toarray()

tree = DecisionTreeClassifier(max_depth = 5)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("tree accuracy= {}".format(accuracy_score(y_test, y_pred)))

m, n = X.shape
forest = RandomForestClassifier(max_depth = 5, num_trees = 100, 
    feature_sample_rate = 1.0 / np.sqrt(n), data_sample_rate = 0.2)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print("forest accuracy= {}".format(accuracy_score(y_test, y_pred)))




































