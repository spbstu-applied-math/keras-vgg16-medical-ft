import tensorflow as tf
import json
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from numba import njit
from io import BytesIO
from PIL import Image

print(device_lib.list_local_devices())

train_path = 'TUPD/train'
valid_path = 'TUPD/valid'
test_path = 'TUPD/test'

train_batches= ImageDataGenerator(rescale=1. / 255, preprocessing_function = tf.keras.applications.vgg16.preprocess_input) \
.flow_from_directory(directory = train_path, target_size = (224,224), classes = ['CNV','DME','DRUSEN','NORMAL'], batch_size = 10)

test_batches= ImageDataGenerator(rescale=1. / 255, preprocessing_function = tf.keras.applications.vgg16.preprocess_input) \
.flow_from_directory(directory = test_path, target_size = (224,224), classes = ['CNV','DME','DRUSEN','NORMAL'],batch_size = 10, shuffle = True)

valid_batches= ImageDataGenerator(rescale=1. / 255, preprocessing_function = tf.keras.applications.vgg16.preprocess_input) \
.flow_from_directory(directory = valid_path, target_size = (224,224), classes = ['CNV','DME','DRUSEN','NORMAL'],batch_size = 10)

vgg16_model = tf.keras.applications.vgg16.VGG16(include_top = False, input_shape=(224, 224, 3))

#vgg16_model.summary()

model = Sequential()
for layer in vgg16_model.layers:
  model.add(layer)

#model.summary()

for layer in model.layers[:-1]:
   layer.trainable = False

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(units = 4, activation = 'softmax'))
model.summary()

model.compile(optimizer = Adam(learning_rate = 0.000001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

hist = model.fit(x = train_batches, validation_data = valid_batches, epochs = 20, verbose = 1)


def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


visualize_results(hist)

res = model.evaluate(test_batches)
json.dump(
    obj = {
        'accuracy': res[1]
    },
    fp = open('eval.json', 'w')
)