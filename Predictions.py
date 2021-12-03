# import cv2
import time

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from DecisionTree import TrainModel
from sklearn.metrics import classification_report, confusion_matrix


Categories = {0: '(20km/h)',
              1: '(30km/h)',
              2: '(50km/h)',
              3: '(60km/h)',
              4: '(70km/h)',
              5: '(80km/h)',
              6: 'End of (80km/h)',
              7: '(100km/h)',
              8: '(120km/h)',
              9: 'No passing',
              10: 'No overtaking by trucks',
              11: 'Crossroads ahead side roads to right and left',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'No entry for trucks',
              17: 'No entry',
              18: 'other danger',
              19: 'curve left',
              20: 'curve right',
              21: 'Double curve, first left',
              22: 'uneven road',
              23: 'Slippery road',
              24: 'Road narrows on the right',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild animals crossing',
              32: 'End all prohibitions',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or right',
              37: 'Go straight or left',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout',
              41: 'End of overtaking prohibition',
              42: 'End no overtaking for large vehicles'}

# unsplit_path = "UnsplitData"
# folders = os.listdir(unsplit_path)
# train_number = []
# class_num = []

# for folder in folders:
# train_files = os.listdir(unsplit_path + '/' + folder)
# train_number.append(len(train_files))
# class_num.append(Categories[int(folder)])

# plt.figure(figsize=(16, 30))
# plt.bar(class_num, train_number)
# plt.xticks(class_num, rotation='vertical')
# plt.show()

# Code used to generate the frequency of classes in training set. Output is in the report
# Code inspiration from https://www.kaggle.com/indhusree/traffic-signal-predection-cnn

test_path = "StreetSignModel/Data/StreetSigns/Test"

StreetSignModelVgg16 = load_model("Models/MLModelVGG16.h5")
StreetSignMobileNet = load_model("Models/StreetSignMobileNet.h5")

VGG16test_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(directory=test_path,
                                                                                                    target_size=(
                                                                                                    224, 224),
                                                                                                    classes=[f'{n}' for
                                                                                                             n in
                                                                                                             range(43)],
                                                                                                    batch_size=32,
                                                                                                    shuffle=False)

MobileNetTestBatches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), batch_size=32, shuffle=False)

print("Starting evaluation for ML Model VGG16")
starttime = time.time()
VGG16Score = StreetSignModelVgg16.evaluate(VGG16test_batches)
VGG16EvaluationTime = time.time()-starttime
print("Evaluation of ML Model VGG16 Complete")

print("Starting predictions for ML Model VGG16")
starttime = time.time()
y_pred = np.argmax(StreetSignModelVgg16.predict(VGG16test_batches), axis=-1)
VGG16PredictionTime = time.time() - starttime
print("Predictions for ML Model VGG16 Complete, Total Time: ", VGG16PredictionTime, "Seconds")

print(" ")
print(" ")
print(" ")

con_mat = tf.math.confusion_matrix(labels=VGG16test_batches.labels, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=Categories, columns=Categories)
figure = plt.figure(figsize=(16, 12))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# Used code from a website instructing on how to use seaborn to create a confusion matrix with a heatmap.
# Cannot find link


print("Starting evaluation for ML Model MobileNet")
starttime = time.time()
MobileNetScore = StreetSignMobileNet.evaluate(MobileNetTestBatches)
MobileNetEvaluationTime = time.time() - starttime
print("Evaluation of ML Model ML Model MobileNet Complete")

print("Starting Predictions for MobileNet Model: ")
starttime = time.time()
y_pred = np.argmax(StreetSignMobileNet.predict(MobileNetTestBatches), axis=-1)
MobileNetPredictionTime = time.time() - starttime
print("Finished predictions for MobileNet Model. Total Time: ", MobileNetPredictionTime, "Seconds")

con_mat = tf.math.confusion_matrix(labels=MobileNetTestBatches.labels, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=Categories, columns=Categories)
figure = plt.figure(figsize=(16, 12))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# Used code from a website instructing on how to use seaborn to create a confusion matrix with a heatmap.
# Cannot find link


CummulativeResults = [
    ["VGG 16", VGG16Score[0], VGG16Score[1], VGG16EvaluationTime, VGG16PredictionTime],
    ["MobileNet", MobileNetScore[0], MobileNetScore[1], MobileNetEvaluationTime, MobileNetPredictionTime]
]

print("Final Results: ")
print(tabulate(CummulativeResults, headers=["Model Name", "Final Loss", "Test Accuracy", "Total evaluation time ", "Total Prediction time"]))


df_test1 = pd.read_csv('Data/StreetSigns/csvfiles/Test.csv')
df_test1.head()

df_test = df_test1.merge(df_meta, on="ClassId")

feature_cols = ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ShapeId', 'ColorId']

X_test = df_test[feature_cols]
y_test = df_test.ClassId

dt_model = joblib.load('DecisionTree_model.sav')

starttime = time.time()
y_pred = clf.predict(X_test)
dt_prediction_time = time.time() - starttime
print("Prediction Time for decision Tree: ", dt_prediction_time, "s")

print(classification_report(y_test, y_pred))

dt_confusion_matrix = confusion_matrix(y_test, y_pred)


