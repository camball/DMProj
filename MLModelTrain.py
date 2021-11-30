import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.callbacks import EarlyStopping

VGG16_default = tf.keras.applications.vgg16.VGG16()

VGG16_default.summary()

StreetSignModel = Sequential()

for layer in VGG16_default.layers[:-1]:
    StreetSignModel.add(layer)

for layer in StreetSignModel.layers:
    layer.trainable = False

StreetSignModel.add(Dense(units=43, activation='softmax'))

StreetSignModel.summary()

train_path = "StreetSignModel/Data/StreetSigns/Train"

valid_path = "StreetSignModel/Data/StreetSigns/Valid"


train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,
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
                                                                                             batch_size=128)

valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,
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
                                                                                             batch_size=128)


imgs, labels = next(train_batches)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(imgs)
print(labels)

StreetSignModel.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4,
                        verbose=1, mode='auto', restore_best_weights=True)

History = StreetSignModel.fit(x=train_batches, validation_data=valid_batches, callbacks=[monitor], epochs=100,
                              verbose=2)

StreetSignModel.save("Models/MLModelVGG16.h5")

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Street Sign Classification Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(History.history['val_loss'])
plt.plot(History.history['loss'])
plt.title('Street Sign Classification Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Plot


