"""ecen5003_lab1.py

# Matt Hartnett's Lab 1 for ECEN5003, Embedded AI

The purpose of this is to learn how to train a basic model in tensorflow. All code here was written by Matt Hartnett, except in specific instances called out in comments.

This code was heavily inspired by [this project](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r)

This code was formatted in such a way to run locally in a VSCode window using the interactive python window and UV as the virtual environment manager.

"""

# %%
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

DIR = "hello_world/"
if not os.path.exists(DIR):
    os.mkdir(DIR)
TF = DIR + "hello_world"
NO_QUANT_TFLITE = DIR + "hello_world_no_quant.tflite"
TFLITE = DIR + "hello_world.tflite"
TFLITE_MICRO = DIR + "hello_world.cc"


# %%
# Always use 42 as seed (answers a lot of questions)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

N_SAMPLES = 1000
NOISE_SCALE = 0.1

# Random set of x values
x = 2 * math.pi * np.random.rand(N_SAMPLES)
# Y values are perfect sine plus normal distribution of noise
y = np.sin(x) + NOISE_SCALE * np.random.randn(N_SAMPLES)

plt.plot(x, y, "b.")
plt.show()

# %%

TRAIN_SPLIT = int(0.6 * N_SAMPLES)
TEST_SPLIT = int(0.2 * N_SAMPLES + TRAIN_SPLIT)

x_train, x_test, x_validate = np.split(x, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y, [TRAIN_SPLIT, TEST_SPLIT])

assert len(x_train) + len(x_test) + len(x_validate) == N_SAMPLES

# %%
model = tf.keras.Sequential()
# 1 -> 8 -> 1
model.add(keras.layers.Dense(8, activation="relu", input_shape=(1,)))
model.add(keras.layers.Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# %%
history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=64,
    validation_data=(x_validate, y_validate),
)
# %%
# This code was copied directly from here: https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r
SKIP = 50
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, "g.", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# This code was copied directly from here: https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r

train_mae = history.history["mae"]
val_mae = history.history["val_mae"]

plt.plot(epochs[SKIP:], train_mae[SKIP:], "g.", label="Training MAE")
plt.plot(epochs[SKIP:], val_mae[SKIP:], "b.", label="Validation MAE")
plt.title("Training and validation mean absolute error")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()

# %%
# This code was copied directly from here: https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r

test_loss, test_mae = model.evaluate(x_test, y_test)

y_test_pred = model.predict(x_test)

plt.clf()
plt.title("Comparison of predictions and actual values")
plt.plot(x_test, y_test, "b.", label="Actual values")
plt.plot(x_test, y_test_pred, "r.", label="TF predictions")
plt.legend()
plt.show()
