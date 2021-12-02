from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


VGG16_default = VGG16()

VGG16_default.summary()

StreetSignModel = Sequential()

for layer in VGG16_default.layers[:-1]:
    StreetSignModel.add(layer)

for layer in StreetSignModel.layers[:-10]:
    layer.trainable = False

StreetSignModel.add(Dense(units=43, activation='softmax'))

StreetSignModel.summary()

train_path = "StreetSignModel/Data/StreetSigns/Train"
valid_path = "StreetSignModel/Data/StreetSigns/Valid"
classes = [f'{n}' for n in range(43)]

train_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(directory=train_path,
                                                                                                target_size=(224, 224),
                                                                                                classes=classes,
                                                                                                batch_size=32)

valid_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(directory=valid_path,
                                                                                                target_size=(224, 224),
                                                                                                classes=classes,
                                                                                                batch_size=32)



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


