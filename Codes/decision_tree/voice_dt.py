import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import OneHotEncoder 
from machine_learning.decision_tree.lib.decision_tree_classifier import DecisionTreeClassifier

def get_data():
    df = pd.read_csv("/Users/wanglei/ML_data/decision_tree/voice.csv")
    y = (df["label"].values=="male").astype(np.int)
    df.drop(['label'], 1, inplace = True)
    X = df.values
    return X, y.reshape(-1,1)

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
model = DecisionTreeClassifier(max_depth = 3)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train).toarray()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy = {}".format(accuracy))





























