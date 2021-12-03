import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import timeit

#timer start
start = timeit.default_timer()

#merge meta and train dataset to increase accuracy
df_train = pd.read_csv('StreetSignModel/Data/StreetSigns/csvfiles/Train.csv')
df_meta = pd.read_csv('StreetSignModel/Data/StreetSigns/csvfiles/Meta.csv')

df_train.head()
df_meta.head()

#merge meta and train dataset to increase accuracy
df = df_train.merge(df_meta, on="ClassId")

feature_cols = ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ShapeId', 'ColorId']

X_train = df[feature_cols]
y_train = df.ClassId

#create decision tree training model
clf = DecisionTreeClassifier(max_depth=9, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)

#store model
filename = 'Models/DecisionTree_model.sav'
joblib.dump(clf, filename)

#timer end
stop = timeit.default_timer()

print(f'Decision Tree Model trained in {round(stop - start, 2)} seconds')
