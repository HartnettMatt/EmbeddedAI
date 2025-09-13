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
TF = DIR + "model"
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
x = 2 * math.pi * np.random.rand(N_SAMPLES).astype(np.float32)
# Y values are perfect sine plus normal distribution of noise
y = np.sin(x) + NOISE_SCALE * np.random.randn(N_SAMPLES).astype(np.float32)

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
# 1 -> 8 -> 8 -> 1
model.add(keras.layers.Dense(8, activation="relu", input_shape=(1,)))
model.add(keras.layers.Dense(16, activation="relu"))
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

# %%

# This code was copied directly from here: https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r

# Draw a graph of the loss, which is the distance between
# the predicted and actual values during training and validation.
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(train_loss) + 1)

# Exclude the first few epochs so the graph is easier to read
SKIP = 100

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)

plt.plot(epochs[SKIP:], train_loss[SKIP:], "g.", label="Training loss")
plt.plot(epochs[SKIP:], val_loss[SKIP:], "b.", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)

# Draw a graph of mean absolute error, which is another way of
# measuring the amount of error in the prediction.
train_mae = history.history["mae"]
val_mae = history.history["val_mae"]

plt.plot(epochs[SKIP:], train_mae[SKIP:], "g.", label="Training MAE")
plt.plot(epochs[SKIP:], val_mae[SKIP:], "b.", label="Validation MAE")
plt.title("Training and validation mean absolute error")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()

# %%
model.export(TF)
converter = tf.lite.TFLiteConverter.from_saved_model(TF)
model_no_quant_tflite = converter.convert()

open(NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)


def representative_dataset():
    for i in range(500):
        yield ([x_train[i].reshape(1, 1)])


# This code was copied directly from here: https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r
# Set the optimization flag.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
open(TFLITE, "wb").write(model_tflite)


# %%
# This code was copied directly from here: https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r
def predict_tflite(tflite_model, x_test):
    # Prepare the test data
    x_test_ = x_test.copy()
    x_test_ = x_test_.reshape((x_test.size, 1))
    x_test_ = x_test_.astype(np.float32)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # If required, quantize the input layer (from float to integer)
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        x_test_ = x_test_ / input_scale + input_zero_point
        x_test_ = x_test_.astype(input_details["dtype"])

    # Invoke the interpreter
    y_pred = np.empty(x_test_.size, dtype=output_details["dtype"])
    for i in range(len(x_test_)):
        interpreter.set_tensor(input_details["index"], [x_test_[i]])
        interpreter.invoke()
        y_pred[i] = interpreter.get_tensor(output_details["index"])[0]

    # If required, dequantized the output layer (from integer to float)
    output_scale, output_zero_point = output_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        y_pred = y_pred.astype(np.float32)
        y_pred = (y_pred - output_zero_point) * output_scale

    return y_pred


def evaluate_tflite(tflite_model, x_test, y_true):
    global model
    y_pred = predict_tflite(tflite_model, x_test)
    loss_function = tf.keras.losses.get(model.loss)
    loss = loss_function(y_true, y_pred).numpy()
    return loss


# %%
# This code was copied directly from here: https://colab.research.google.com/github/tensorflow/tflite-micro/blob/18aec279a0f35af82e4543feae00e1c87a75c8bf/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=CmvA-ksoln8r
# Calculate predictions
y_test_pred_tf = model.predict(x_test)
y_test_pred_no_quant_tflite = predict_tflite(model_no_quant_tflite, x_test)
y_test_pred_tflite = predict_tflite(model_tflite, x_test)
# Compare predictions
plt.clf()
plt.title("Comparison of various models against actual values")
plt.plot(x_test, y_test, "bo", label="Actual values")
plt.plot(x_test, y_test_pred_tf, "ro", label="TF predictions")
plt.plot(x_test, y_test_pred_no_quant_tflite, "bx", label="TFLite predictions")
plt.plot(x_test, y_test_pred_tflite, "gx", label="TFLite quantized predictions")
plt.legend()
plt.show()
# Calculate loss
loss_tf, _ = model.evaluate(x_test, y_test, verbose=0)
loss_no_quant_tflite = evaluate_tflite(model_no_quant_tflite, x_test, y_test)
loss_tflite = evaluate_tflite(model_tflite, x_test, y_test)
# Compare loss
df = pd.DataFrame.from_records(
    [
        ["TensorFlow", loss_tf],
        ["TensorFlow Lite", loss_no_quant_tflite],
        ["TensorFlow Lite Quantized", loss_tflite],
    ],
    columns=["Model", "Loss/MSE"],
    index="Model",
).round(4)
df
# Calculate size
size_tf = os.path.getsize(TF)
size_no_quant_tflite = os.path.getsize(NO_QUANT_TFLITE)
size_tflite = os.path.getsize(TFLITE)
# Compare size
pd.DataFrame.from_records(
    [
        ["TensorFlow", f"{size_tf} bytes", ""],
        [
            "TensorFlow Lite",
            f"{size_no_quant_tflite} bytes ",
            f"(reduced by {size_tf - size_no_quant_tflite} bytes)",
        ],
        [
            "TensorFlow Lite Quantized",
            f"{size_tflite} bytes",
            f"(reduced by {size_no_quant_tflite - size_tflite} bytes)",
        ],
    ],
    columns=["Model", "Size", ""],
    index="Model",
)
