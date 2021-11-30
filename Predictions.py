import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.callbacks import EarlyStopping

CATEGORIES = ["20", "30", "50", "60", "70", "80", "80 Ends", "100", "120", "No overtaking", "No overtaking by trucks",
              "Crossroads ahead, side roads to right and left", "Priority Road", "Yield", "Stop", "No Vehicles",
              "No entry for trucks", "Do not enter", "Other danger", "Curve to the left", "Curve to the right",
              "Double curve, first left", "Uneven road", "Slippery road", "Road Narrow on right", "Road work",
              "Traffic signals", "Pedestrian crossing", "Children crossing", "Cyclist crossing",
              "Crossroads without priority", "Wild animal crossing", "End all prohibitions", "Turn right", "Turn left",
              "Straight ahead", "Straight ahead or turn right", "Straight ahead or turn left", "Keep right", "Keep left"
    , "Roundabout", "Ends overtaking prohibition", "Ends overtaking prohibition for large vehicles"
              ]

StreetSignModelVgg16 = load_model("Models/MLModelVGG16.h5")
test_path = "StreetSignModel/Data/StreetSigns/Test"
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
                                                                                             target_size=(224, 224),
                                                                                             classes=['0', '1', '2',
                                                                                                      '3', '4', '5',
                                                                                                      '6', '7', '8',
                                                                                                      '9', '10', '11',
                                                                                                      '12', '13', '14',
                                                                                                      '15', '16', '17',
                                                                                                      '18', '19', '20',
                                                                                                      '21', '22', '23',
                                                                                                      '24', '25', '26',
                                                                                                      '27', '28', '29',
                                                                                                      '30', '31', '32',
                                                                                                      '33', '34', '35',
                                                                                                      '36', '37', '38',
                                                                                                      '39', '40', '41',
                                                                                                      '42'],
                                                                                             batch_size=32,
                                                                                             shuffle=False)

Score = StreetSignModelVgg16.evaluate(test_batches)
