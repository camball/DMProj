import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import timeit

start = timeit.default_timer()

df_train = pd.read_csv('Data/StreetSigns/csvfiles/Train.csv')
df_test1 = pd.read_csv('Data/StreetSigns/csvfiles/Test.csv')
df_meta = pd.read_csv('Data/StreetSigns/csvfiles/Meta.csv')

df_train.head()
df_test1.head()
df_meta.head()

df = df_train.merge(df_meta, on="ClassId")
df_test = df_test1.merge(df_meta, on="ClassId")

feature_cols = ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ShapeId', 'ColorId']

X_train = df[feature_cols]
y_train = df.ClassId

X_test = df_test[feature_cols]
y_test = df_test.ClassId

clf = DecisionTreeClassifier(max_depth=9, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)

stop = timeit.default_timer()

print(f'Decision Tree Model trained in {round(stop - start, 2)} seconds')
