import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from matplotlib import pyplot as plt

df_train = pd.read_csv('Data/StreetSigns/csvfiles/Train.csv')
df_test1 = pd.read_csv('Data/StreetSigns/csvfiles/Test.csv')
df_meta = pd.read_csv('Data/StreetSigns/csvfiles/Meta.csv')

df_train.head()
df_test1.head()
df_meta.head()

df = df_train.merge(df_meta, on="ClassId")
df_test = df_test1.merge(df_meta, on="ClassId")

feature_cols = ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ShapeId', 'ColorId']
features = ['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']

X_train = df[feature_cols]
y_train = df.ClassId

X_test = df_test[feature_cols]
y_test = df_test.ClassId

clf = DecisionTreeClassifier(max_depth=9, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

error_rate = 1 - accuracy_score(y_test, y_pred)
print('Error rate', error_rate)
