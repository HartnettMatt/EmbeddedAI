##
## File: part1.py
## Author: Matt Hartnett
## Simple audio recognition based on this example: https://www.tensorflow.org/tutorials/audio/simple_audio
##

# %%
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from glob import glob

# %%
# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# %%
# Download data

data_dir = pathlib.Path(__file__).parent.resolve() / "../testdata/mini_speech_commands/"

if not data_dir.exists():
    os.mkdir(data_dir)
if not any(data_dir.iterdir()):
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=".",
        cache_subdir=data_dir,
    )
else:
    print(f"Files found at {data_dir}, not downloading new files")

data_dir = data_dir / "mini_speech_commands_extracted/mini_speech_commands"
matt_data_dir = pathlib.Path(__file__).parent.resolve() / "../testdata/"

# %%
# Find data files
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != "README.md") & (commands != ".DS_Store")]
print("Commands:", commands)

# %%
# Pull data into keras
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset="both",
)

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)


train_ds.element_spec


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


# Clean out extra axis
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Break into two sets
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)

# %%
# Plot Data

label_names[[1, 1, 3, 0]]
plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
    plt.subplot(rows, cols, i + 1)
    audio_signal = example_audio[i]
    plt.plot(audio_signal)
    plt.title(label_names[example_labels[i]])
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])


# %%
# Spectrogram function defintions
# Constants to match C++:
ROWS = 49
COLUMNS = 40
WINDOW = 30e-3
STRIDE = 20e-3
SAMPLE_RATE = 16e3

# Derived:
FRAME_LENGTH = int(WINDOW * SAMPLE_RATE)
FRAME_STEP = int(STRIDE * SAMPLE_RATE)
FFT_LENGTH = 2 * int(COLUMNS - 1)


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


# %%
# Explore data
for i in range(3):
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)

    print("Label:", label)
    print("Waveform shape:", waveform.shape)
    print("Spectrogram shape:", spectrogram.shape)
    print("Audio playback")
    display.display(display.Audio(waveform, rate=16000))

    # Plot data
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title("Waveform")
    axes[0].set_xlim([0, 16000])

    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title("Spectrogram")
    plt.suptitle(label.title())
    plt.show()


# %%
# Graph multiple spectrograms:
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break
rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])

plt.show()

# %%
# Train model
# Cache and prefetch dataset
train_spectrogram_ds = (
    train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)


# Flatten dataset
def flatten_ds(spec_ds):
    return spec_ds.map(
        lambda spec, lab: (
            tf.reshape(tf.cast(spec, tf.float32), [-1, ROWS * COLUMNS]),
            lab,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


train_flat = (
    flatten_ds(train_spectrogram_ds).cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
)
val_flat = flatten_ds(val_spectrogram_ds).cache().prefetch(tf.data.AUTOTUNE)
test_flat = flatten_ds(test_spectrogram_ds).cache().prefetch(tf.data.AUTOTUNE)

# Create convolutional neural network
input_shape = (ROWS * COLUMNS,)
print("Input shape:", input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(train_flat.map(lambda x, _: x))

model = models.Sequential(
    [
        layers.Input(shape=input_shape),
        norm_layer,
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.15),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_labels),  # logits; no softmax needed on-device
    ]
)

model.summary()

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
EPOCHS = 30
history = model.fit(
    train_flat,
    validation_data=val_flat,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

# %%
# Plot loss
metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
plt.legend(["loss", "val_loss"])
plt.ylim([0, max(plt.ylim())])
plt.xlabel("Epoch")
plt.ylabel("Loss [CrossEntropy]")

plt.subplot(1, 2, 2)
plt.plot(
    history.epoch,
    100 * np.array(metrics["accuracy"]),
    100 * np.array(metrics["val_accuracy"]),
)
plt.legend(["accuracy", "val_accuracy"])
plt.ylim([0, 100])
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")

# %%
# Evaluate the model
model.evaluate(test_flat, return_dict=True)
y_pred = model.predict(test_flat)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_flat.map(lambda s, lab: lab)), axis=0)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx, xticklabels=label_names, yticklabels=label_names, annot=True, fmt="g"
)
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.show()

# %%
# Evaluate the model with matt data:
matt_wavs = sorted(glob(str(matt_data_dir / "*.wav")))
matt_filenames = [os.path.basename(wav) for wav in matt_wavs]
matt_labels = [file.split("_")[1] for file in matt_filenames]

name_to_idx = {name: i for i, name in enumerate(label_names)}
label_ids = [name_to_idx[lbl] for lbl in matt_labels if lbl in name_to_idx]
matt_wavs = [w for w, lbl in zip(matt_wavs, matt_labels) if lbl in name_to_idx]

paths_ds = tf.data.Dataset.from_tensor_slices(matt_wavs)
labels_ds = tf.data.Dataset.from_tensor_slices(label_ids)
matt_ds = tf.data.Dataset.zip((paths_ds, labels_ds))


def load_and_flatten(path, lab):
    audio_bytes = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(
        audio_bytes, desired_channels=1, desired_samples=16000
    )
    audio = tf.squeeze(audio, axis=-1)
    spec = get_spectrogram(audio)
    spec = tf.cast(spec, tf.float32)
    spec = tf.reshape(spec, [ROWS * COLUMNS])
    return spec, lab


matt_flat = (
    matt_ds.map(load_and_flatten, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(64)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

model.evaluate(matt_flat, return_dict=True)

y_logits = model.predict(matt_flat)
y_pred = tf.argmax(y_logits, axis=1)

# True labels
y_true = tf.concat(list(matt_flat.map(lambda s, lab: lab)), axis=0)
y_true_int = tf.cast(y_true, tf.int32)

# --- Soft confusion matrix (average confidence per true class) ---
# This section was written by Chat GPT 5-Thinking with the prompt:
# Modify this confusion matrix to include confidence level in each guess instead of single value reporting
probs = tf.nn.softmax(y_logits, axis=1)  # [N, num_labels]
soft_sum = tf.math.unsorted_segment_sum(
    probs, y_true_int, num_segments=len(label_names)
)
row_counts = tf.math.bincount(
    y_true_int, minlength=len(label_names), maxlength=len(label_names)
)
row_counts = tf.cast(tf.maximum(row_counts, 1), tf.float32)  # avoid div-by-zero
confusion_mtx = soft_sum / row_counts[:, None]  # [num_labels, num_labels]
# End of Chat GPT code

plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx,
    xticklabels=label_names,
    yticklabels=label_names,
    annot=True,
    fmt=".3f",
)
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.show()
# %%
# Run inference
SAMPLE = "go"
x = matt_data_dir / f"matt_{SAMPLE}_1000ms.wav"
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(
    x,
    desired_channels=1,
    desired_samples=16000,
)
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis, ...]

# Make spectrogram exactly like training, then FLATTEN
spec = tf.cast(x, tf.float32)
spec = tf.reshape(spec, [1, ROWS * COLUMNS])  # [1, 1960]

prediction = model(spec, training=False)
x_labels = label_names
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title(SAMPLE)
plt.show()

display.display(display.Audio(waveform, rate=16000))


# %%
# Export the model
EXPORT_DIR = pathlib.Path(__file__).parent.resolve() / "../models"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXPORT_DIR / "matt_micro_speech_quantized.tflite"


def _rep_data():
    # Small, fast representative set for calibration; adjust 'take' if you want tighter scales
    for specs, _ in train_spectrogram_ds.take(20):
        for i in range(min(8, specs.shape[0])):
            yield [tf.cast(specs[i : i + 1], tf.float32)]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = _rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8

tflite_int8 = conv.convert()
OUT_PATH.write_bytes(tflite_int8)
print(f"[export] wrote {OUT_PATH} ({len(tflite_int8)} bytes)")
