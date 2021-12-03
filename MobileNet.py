import keras.applications.mobilenet_v3
from keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

train_path = "StreetSignModel/Data/StreetSigns/Train"
valid_path = "StreetSignModel/Data/StreetSigns/Valid"

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), batch_size=32)
valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), batch_size=32 )

mobile = keras.applications.mobilenet.MobileNet()

mobile.summary()
x = mobile.layers[-2].output
output = Dense(units=43, activation='softmax')(x)
StreetSignMobileNet = Model(inputs=mobile.input, outputs=output)
StreetSignMobileNet.summary()

for layer in StreetSignMobileNet.layers[:-10]:
    layer.trainable = False


monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4,
                        verbose=1, mode='auto', restore_best_weights=True)


StreetSignMobileNet.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=["accuracy"])

history = StreetSignMobileNet.fit(x=train_batches, validation_data=valid_batches, epochs=30, verbose=2)

StreetSignMobileNet.save("Models/StreetSignMobileNet.h5")

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