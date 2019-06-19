from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from machine_learning.decision_tree.lib.decision_tree_classifier import DecisionTreeClassifier
from machine_learning.decision_tree.lib.random_forest_classifier import RandomForestClassifier
from sklearn.metrics import accuracy_score


X, y = make_blobs(n_samples=1000, centers=3, random_state=0, cluster_std=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("decision tree accuracy= {}".format(accuracy_score(y_test, y_pred)))

forest = RandomForestClassifier(max_depth=1, num_trees=100, feature_sample_rate=0.5, data_sample_rate=0.1)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print("random forest accuracy= {}".format(accuracy_score(y_test, y_pred)))



















