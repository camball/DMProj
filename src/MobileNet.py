import keras.applications.mobilenet_v3
from keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

train_path = "../StreetSigns/Train"  # Declares the training path
valid_path = "../StreetSigns/Valid"  # Declares the validation path

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), batch_size=32)
# Generates training and validation batches. Uses the MobileNet preprocessing function.
# The Preprocessing function scales the pixel values to between -1 and 1

valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), batch_size=32)

mobile = keras.applications.mobilenet.MobileNet()  # Creates an instance of the MobileNet ML Model

x = mobile.layers[-2].output  # Copies all but the last layer of MobileNet to x

output = Dense(units=43, activation='softmax')(x)  # Places a Dense layer with 43 output nodes at the end of the outputs
StreetSignMobileNet = Model(inputs=mobile.input, outputs=output)  # Creates a new Functional model with all the
# inputs of MobileNet, but with its output replaced with a Dense Layer with 43 nodes


for layer in StreetSignMobileNet.layers[:-10]:
    layer.trainable = False  # Freezes all but the last 10 layers to remain consistent with VGG16 Model for comparison

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4,
                        verbose=1, mode='auto', restore_best_weights=True)
# Checks to see if val_loss is decreasing by at least .0001 per round. If not the model has 3 chances to correct
# that before it is prematurely terminated. This is to prevent overfitting

StreetSignMobileNet.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=["accuracy"])


history = StreetSignMobileNet.fit(x=train_batches, validation_data=valid_batches, epochs=30, verbose=2)
# Trains the model for a maximum of 30 epochs.

StreetSignMobileNet.save("../Models/StreetSignMobileNet.h5")
# Saves the model to the models folder.

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Street Sign Classification Model Accuracy - Mobile Net')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Street Sign Classification Model Loss - Mobile Net')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
