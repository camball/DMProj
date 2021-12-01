# import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

CATEGORIES = ["20", "30", "50", "60", "70", "80", "80 Ends", "100", "120", "No overtaking", "No overtaking by trucks",
              "Crossroads ahead, side roads to right and left", "Priority Road", "Yield", "Stop", "No Vehicles",
              "No entry for trucks", "Do not enter", "Other danger", "Curve to the left", "Curve to the right",
              "Double curve, first left", "Uneven road", "Slippery road", "Road Narrow on right", "Road work",
              "Traffic signals", "Pedestrian crossing", "Children crossing", "Cyclist crossing",
              "Crossroads without priority", "Wild animal crossing", "End all prohibitions", "Turn right", "Turn left",
              "Straight ahead", "Straight ahead or turn right", "Straight ahead or turn left", "Keep right", "Keep left", 
              "Roundabout", "Ends overtaking prohibition", "Ends overtaking prohibition for large vehicles"]

StreetSignModelVgg16 = load_model("Models/MLModelVGG16.h5")
test_path = "StreetSignModel/Data/StreetSigns/Test"

test_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory( directory=test_path, 
                                                                                                target_size=(224, 224),
                                                                                                classes=[f'{n}' for n in range(43)],
                                                                                                batch_size=32,
                                                                                                shuffle=False)

Score = StreetSignModelVgg16.evaluate(test_batches)
