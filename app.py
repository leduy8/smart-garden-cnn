from plot_confusion_matrix import plot_confusion_matrix

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

# * Create models dir if not exists
from pathlib import Path
Path("./models").mkdir(parents=True, exist_ok=True)

# * Disable GPU for tensorflow
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# * Using GPU for processing
physical_devices = tf.config.experimental.list_physical_devices("GPU")
#print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mobile = tf.keras.applications.mobilenet.MobileNet()

# * Process the data
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), batch_size=10)
# ? In test batches, we set shuffle to False, so that we can access the corresponding non-shuffle test labels to plot in the confustion matrix
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), batch_size=10, shuffle=False)

# ******************* CREATE MODEL TO TRAIN   ******************* #
# * Grab layers from start at the last sixth layer in mobilenet model (i.e grab all except last 5 layers in mobilenet)
x = mobile.layers[-6].output

# * Create an output layer
# ? This is Keras Functional API syntax, this is chain layer calls to specify the model's forward pass
output = Dense(units=3, activation='softmax')(x)

# * Create the fine-tuned model
model = Model(inputs=mobile.input, outputs=output)

# * Freeze the weight, only train last 23 layer for new dataset
# ? 23 is not a magic number, it's just an experiment and find it work pretty decently
for layer in model.layers[:-23]:
    layer.trainable = False

# model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, steps_per_epoch=len(train_batches),
          validation_data=valid_batches, validation_steps=len(valid_batches), epochs=3, verbose=2)

# * Save model(the architecture, the weights, the optimizer, the state of the optimizer, the learning rate, the loss, etc.) to a .h5 file
# ? If found a model, delete it and save a new one
# if os.path.isfile("models/apple_leaves_diseases_model.h5") is True:
#     os.remove("models/apple_leaves_diseases_model.h5")
model.save(r"models/apple_leaves_diseases_model.h5")

# ******************* LOAD MODEL AFTER TRAINED   ******************* #
# * Load model
# model = load_model(r'models/apple_leaves_diseases_model.h5')
# * Freeze the weight, only train last 23 layer for new dataset
# ? 23 is not a magic number, it's just an experiment and find it work pretty decently
# for layer in model.layers[:-23]:
#     layer.trainable = False

# ******************* LET MODEL PREDICTS ON TEST DATA   ******************* #
# * Get unshuffle test labels
test_labels = test_batches.classes

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
cm_plot_labels = ['healthy', 'rust', 'scab']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
