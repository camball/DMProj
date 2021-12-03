import time
import timeit

import joblib
import numpy as np
import pandas as pd
from tensorflow import math
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16, mobilenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
# from sklearn.metrics import classification_report, confusion_matrix


Categories = {
    0: '(20km/h)',
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
    42: 'End no overtaking for large vehicles'
}

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

test_path = "StreetSignModel/Data/StreetSigns/Test" # Test path for the models

StreetSignModelVgg16 = load_model("Models/MLModelVGG16.h5")  # Loads in the VGG16 Based Sequential model
StreetSignMobileNet = load_model("Models/StreetSignMobileNet.h5")  # Loads in the MobileNet Based Functional Model

# Creates the test batches for VGG16, using VGG16 preprocessing. Avoids shuffling labels to create Confusion Matrix
VGG16test_batches = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
VGG16test_batches.flow_from_directory(
    directory=test_path, 
    target_size=(224, 224),
    classes=[f'{n}' for n in range(43)],
    batch_size=32,
    shuffle=False
)

# Creates the test batches for MobileNet, uses MobileNet Preprocessing
MobileNetTestBatches = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)
MobileNetTestBatches.flow_from_directory(
    directory=test_path, 
    target_size=(224, 224), 
    batch_size=32, 
    shuffle=False
)

# Creates the test batches for MobileNet, uses MobileNet Preprocessing


print("Starting evaluation for ML Model VGG16")
starttime = time.time()
VGG16Score = StreetSignModelVgg16.evaluate(VGG16test_batches)  # Evaluates the VGG16 Model Accuracy and Loss
VGG16EvaluationTime = time.time() - starttime
print("Evaluation of ML Model VGG16 Complete")

print("Starting predictions for ML Model VGG16")
starttime = time.time()
y_pred = np.argmax(StreetSignModelVgg16.predict(VGG16test_batches), axis=-1)  # Gets the top prediction for every image
VGG16PredictionTime = time.time() - starttime
print("Predictions for ML Model VGG16 Complete, Total Time: ", VGG16PredictionTime, "Seconds")

print(" ")
print(" ")
print(" ")

# Create Confusion Matrix Heatmap to see where the model is misclassified images
con_mat = math.confusion_matrix(labels=VGG16test_batches.labels, predictions=y_pred).numpy()
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
MobileNetScore = StreetSignMobileNet.evaluate(MobileNetTestBatches)  # Evaluate MobileNet Model Accuracy and Loss
MobileNetEvaluationTime = time.time() - starttime
print("Evaluation of ML Model ML Model MobileNet Complete")

print("Starting Predictions for MobileNet Model: ")
starttime = time.time()
y_pred = np.argmax(StreetSignMobileNet.predict(MobileNetTestBatches), axis=-1)  # Get top prediction for every image

MobileNetPredictionTime = time.time() - starttime
print("Finished predictions for MobileNet Model. Total Time: ", MobileNetPredictionTime, "Seconds")

# Create Confusion Matrix Heatmap to see where the model is misclassified images
con_mat = math.confusion_matrix(labels=MobileNetTestBatches.labels, predictions=y_pred).numpy()
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

print(" ")
print(" ")
print(" ")

df_test1 = pd.read_csv('StreetSignModel/Data/StreetSigns/csvfiles/Test.csv')
df_meta = pd.read_csv('StreetSignModel/Data/StreetSigns/csvfiles/Meta.csv')
feature_cols = ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ShapeId', 'ColorId']

df_test1.head()

# Merge test and meta to match training data
df_test = df_test1.merge(df_meta, on="ClassId")

X_test = df_test[feature_cols]
y_test = df_test.ClassId

# Load decision tree model
clf = joblib.load('Models/DecisionTree_model.sav')

print("Starting Evaluation for Decision Tree Model")
starttime = timeit.default_timer()
score = clf.score(X=X_test, y=y_test)
DecisionTreeEvaluationTime = timeit.default_timer() - starttime
print("Finished predictions for Decision Tree Model. Total Time: ", DecisionTreeEvaluationTime, "Seconds")

print("Starting Predictions for Decision Tree Model")
starttime = timeit.default_timer()
y_pred = clf.predict(X_test)
DecisionTreePredictTime = timeit.default_timer() - starttime
print("Finished predictions for Decision Tree Model. Total Time: ", DecisionTreePredictTime, "Seconds")

# Decision tree model prediciton confusion matrix heat map
con_mat = math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=Categories, columns=Categories)
figure = plt.figure(figsize=(16, 12))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



CumulativeResults = [
    ["VGG 16", VGG16Score[0], VGG16Score[1], VGG16EvaluationTime, VGG16PredictionTime],
    ["MobileNet", MobileNetScore[0], MobileNetScore[1], MobileNetEvaluationTime, MobileNetPredictionTime],
    ["Decision Tree", "N/A", score, DecisionTreeEvaluationTime, DecisionTreePredictTime]
]

print("Final Results: ")
print(
    tabulate(CumulativeResults, headers=[
            "Model Name", 
            "Final Loss", 
            "Test Accuracy", 
            "Total evaluation time ",
            "Total Prediction time"
        ]
    )
)
