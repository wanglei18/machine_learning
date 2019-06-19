from pandas import read_csv
 
df = read_csv("/Users/wanglei/ML_data/pollution/pollution_series.csv")
values = df.values
n_train_hours = 365 * 24
train = values[:n_train_hours, 1:]
test = values[n_train_hours:, 1:]
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
 
























