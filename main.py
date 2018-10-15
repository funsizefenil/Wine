import pandas as pd
import numpy as np
import data
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

red_data, white_data = data.importer()
y_red = red_data.quality
x_red = red_data.drop('quality', axis = 1)

x_red_train, x_red_test, y_red_train, y_red_test = train_test_split(x_red, y_red, test_size = 0.2, random_state = 123, stratify = y_red)

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv = 10)

clf.fit(x_red_train, y_red_train)

pred = clf.predict(x_red_test)
print (r2_score(y_red_test, pred))
print (mean_squared_error(y_red_test, pred))