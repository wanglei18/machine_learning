import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_data():
    df = pd.read_csv("/Users/wanglei/ML_data/bike.csv")
    df.datetime = df.datetime.apply(pd.to_datetime)
    df['month'] = df.datetime.apply(lambda x: x.month)
    df['hour'] = df.datetime.apply(lambda x: x.hour)
    df['day'] = df.datetime.apply(lambda x: x.day)
    df.drop(['datetime'], 1, inplace = True)
    y = df['count'].values
    df.drop(['casual', 'registered', 'count'], 1, inplace = True)
    feature_list = ['hour', 'season','holiday', 'workingday','weather']
    X = df[feature_list].values
    return X, y
    
X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeRegressor(max_depth = 10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("decision tree r2=", r2_score(y_test, y_pred))

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("linear regression r2=", r2_score(y_test, y_pred))








#plt.show()























