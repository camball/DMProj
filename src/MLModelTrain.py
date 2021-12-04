from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

VGG16_default = VGG16()         # Create instance of the VGG16 ML Model
StreetSignModel = Sequential()  # Create instance of the Sequential Form ML Model with no layers

for layer in VGG16_default.layers[:-1]:
    StreetSignModel.add(layer) # Add every layer from VGG16 to my Sequential Model except the output layer

for layer in StreetSignModel.layers[:-10]:
    layer.trainable = False # Freeze all but last 10 layers. The last 10 remain trainable for our dataset

# Add a Dense (fully connected) layer with 43 output nodes corresponding to our 43 classes
StreetSignModel.add(Dense(units=43, activation='softmax'))


train_path = "../StreetSigns/Train"
valid_path = "../StreetSigns/Valid"

classes = [f'{n}' for n in range(43)] # Each number represents a class

# Preprocess images to prepare them for the model
# The preprocess function subtracts the average RGB Value from every pixel to normalize the images
train_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    classes=classes,
    batch_size=32
)

valid_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory=valid_path,
    target_size=(224, 224),
    classes=classes,
    batch_size=32
)

imgs, labels = next(train_batches) # Grab a batch of images to show what they look like after processing


def plotImages(images_arr):
    _, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show() # Display the batch of images


plotImages(imgs)
print(labels)

# Compile model with Adam Optimizer, an extension of Stochastic Gradient Descent
# Categorical Crossentropy compares predicted probability distribution, with target probability distribution
StreetSignModel.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Checks to see if val_loss is decreasing by at least .0001 per round. If not, the model has 
# 3 chances to correct that before it is prematurely terminated. This is to prevent overfitting
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4,
                        verbose=1, mode='auto', restore_best_weights=True)

# Train the model with a maximum of 100 epochs
History = StreetSignModel.fit(
    x=train_batches, 
    validation_data=valid_batches, 
    callbacks=[monitor], 
    epochs=100,
    verbose=2
)

StreetSignModel.save("../Models/MLModelVGG16.h5")  # Save model architecture and weights

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
