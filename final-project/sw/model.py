# %%
"""
TensorFlow training script for the final project tiny BNN.

The goal is to mimic the hardware accelerator: train on the 8x8 scikit-learn
digits dataset, constrain the network to XNOR-popcount friendly primitives,
and export weights/thresholds plus golden vectors for the RTL testbench.
"""

# %%
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# %%
# Global configuration -------------------------------------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.config.experimental.enable_op_determinism()

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "sw" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
DIGITS_CACHE = DATASET_DIR / "digits.npz"

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
HW_SVH_PATH = REPO_ROOT / "hw" / "model.svh"

BATCH_SIZE = 128
EPOCHS = 250
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_GOLDEN = 200
HIDDEN_UNITS = 128

# %%
# Helper utilities -----------------------------------------------------------
def ste_sign(x: tf.Tensor) -> tf.Tensor:
    """Straight-through estimator for binarization in {+1, -1}."""
    y = tf.sign(x)
    y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
    return x + tf.stop_gradient(y - x)


class BinaryActivation(tf.keras.layers.Layer):
    """Deterministic binarization layer with STE backprop."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        return ste_sign(inputs)

    def get_config(self) -> Dict[str, Any]:  # type: ignore[override]
        return super().get_config()


class BinaryDense(tf.keras.layers.Layer):
    """Dense layer with binarized weights + activations suitable for XNOR logic."""

    def __init__(self, units: int, use_bias: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape: tf.TensorShape) -> None:  # type: ignore[override]
        input_dim = int(input_shape[-1])
        limit = 1.0 / np.sqrt(input_dim)
        self.kernel = self.add_weight(  # type: ignore[attr-defined]
            name="kernel",
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-limit, limit),
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(  # type: ignore[attr-defined]
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        binary_kernel = ste_sign(self.kernel)
        binary_inputs = ste_sign(inputs)
        output = tf.matmul(binary_inputs, binary_kernel)

        if self.bias is not None:
            output = tf.nn.bias_add(output, self.bias)

        return output

    def get_config(self) -> Dict[str, Any]:  # type: ignore[override]
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
            }
        )
        return config


def make_dataset(
    features: np.ndarray, labels: np.ndarray, *, shuffle: bool
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# %%
# Data loading & splitting ---------------------------------------------------
if DIGITS_CACHE.exists():
    cache = np.load(DIGITS_CACHE, allow_pickle=False)
    images_raw = cache["images"]
    labels = cache["target"].astype(np.int32)
    target_names = cache["target_names"]
    print(f"Loaded cached digits dataset from {DIGITS_CACHE}")
else:
    digits = datasets.load_digits()
    images_raw = digits.images
    labels = digits.target.astype(np.int32)
    target_names = digits.target_names
    np.savez(
        DIGITS_CACHE,
        images=images_raw,
        target=labels,
        target_names=target_names,
    )
    print(f"Downloaded scikit-learn digits dataset to {DIGITS_CACHE}")

images = images_raw.astype(np.float32) / 16.0
images = images * 2.0 - 1.0
images = images.reshape(images.shape[0], -1)

holdout = VAL_SPLIT + TEST_SPLIT
X_train, X_hold, y_train, y_hold = train_test_split(
    images,
    labels,
    test_size=holdout,
    random_state=SEED,
    stratify=labels,
)

relative_test = TEST_SPLIT / holdout
X_val, X_test, y_val, y_test = train_test_split(
    X_hold,
    y_hold,
    test_size=relative_test,
    random_state=SEED,
    stratify=y_hold,
)

train_ds = make_dataset(X_train, y_train, shuffle=True)
val_ds = make_dataset(X_val, y_val, shuffle=False)
test_ds = make_dataset(X_test, y_test, shuffle=False)

print(f"Train/Val/Test splits: {len(X_train)}/{len(X_val)}/{len(X_test)} samples")


# %%
# Model definition -----------------------------------------------------------
def build_bnn_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="digit_pixels")
    x = BinaryActivation(name="input_binarize")(inputs)
    x = BinaryDense(HIDDEN_UNITS, use_bias=True, name="bnn_hidden")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, name="bnn_hidden_bn")(x)
    x = BinaryActivation(name="bnn_hidden_act")(x)
    x = tf.keras.layers.Dropout(0.1, name="hidden_dropout")(x)
    x = BinaryDense(num_classes, use_bias=True, name="bnn_logits")(x)
    outputs = tf.keras.layers.Activation("softmax", name="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="digits_bnn")


model = build_bnn_model(input_dim=images.shape[1], num_classes=len(target_names))
model.summary(expand_nested=False, show_trainable=True)


# %%
# Training -------------------------------------------------------------------
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        patience=40,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    ),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
    callbacks=callbacks,
)


# %%
# Evaluation -----------------------------------------------------------------
test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
print("Test metrics:", test_metrics)

test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(test_probs, axis=1)

print("\nClassification report\n")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix\n", confusion_matrix(y_test, y_pred))

metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics["loss"], label="loss")
plt.plot(history.epoch, metrics["val_loss"], label="val_loss")
plt.legend()
plt.ylim([0, max(plt.ylim())])
plt.xlabel("Epoch")
plt.ylabel("Loss [CrossEntropy]")

plt.subplot(1, 2, 2)
plt.plot(
    history.epoch,
    100 * np.array(metrics["acc"]),
    label="accuracy",
)
plt.plot(
    history.epoch,
    100 * np.array(metrics["val_acc"]),
    label="val_accuracy",
)
plt.legend()
plt.ylim([0, 100])
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.tight_layout()
plt.show()

cm = tf.math.confusion_matrix(
    y_test, y_pred, num_classes=len(target_names), dtype=tf.float32
)
cm = cm / tf.reduce_sum(cm, axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm.numpy(), xticklabels=target_names, yticklabels=target_names, annot=True, fmt=".2f")
plt.title("Normalized Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.tight_layout()
plt.show()


# %%
# Export helpers -------------------------------------------------------------
def pack_bits_to_words(bits: np.ndarray, word_width: int = 32) -> List[int]:
    flat = bits.astype(np.uint8).flatten()
    pad = (-len(flat)) % word_width
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    chunks = flat.reshape(-1, word_width)
    words: List[int] = []
    for chunk in chunks:
        word = 0
        for idx, bit in enumerate(chunk):
            word |= int(bit) << idx
        words.append(word)
    return words


def quantize_to_int16(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.rint(arr), -32768, 32767)
    return clipped.astype(np.int16)


def write_svh(mem_words: List[int], depth: int, path: Path) -> None:
    if len(mem_words) < depth:
        mem_words = mem_words + [0] * (depth - len(mem_words))
    elif len(mem_words) > depth:
        mem_words = mem_words[:depth]

    lines: List[str] = []
    lines.append("// Auto-generated model memory")
    lines.append(f"localparam logic [31:0] MODEL_MEM [0:{depth-1}] = '{{")
    for idx, word in enumerate(mem_words):
        sep = "," if idx < depth - 1 else ""
        lines.append(f"    32'h{word:08x}{sep}")
    lines.append("};")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote quantized model -> {path}")


manifest: List[Dict[str, Any]] = []
memh_blobs: List[Dict[str, Any]] = []
layers = model.layers

for idx, layer in enumerate(layers):
    if not isinstance(layer, BinaryDense):
        continue

    weights = layer.kernel.numpy()
    bias = layer.bias.numpy() if layer.bias is not None else np.zeros((layer.units,), dtype=np.float32)
    weights_pm1 = np.where(weights >= 0, 1, -1).astype(np.int8)
    weights_bits = ((weights_pm1 + 1) // 2).astype(np.uint8)

    artifacts: List[Tuple[str, np.ndarray]] = [
        (f"{layer.name}_weights_pm1.npy", weights_pm1),
        (f"{layer.name}_weights_bits.npy", weights_bits),
        (f"{layer.name}_bias.npy", bias.astype(np.float32)),
    ]

    metadata: Dict[str, Any] = {
        "layer": layer.name,
        "input_dim": int(weights.shape[0]),
        "units": layer.units,
        "weights_pm1_file": artifacts[0][0],
        "weights_bits_file": artifacts[1][0],
        "bias_file": artifacts[2][0],
    }

    bn_layer = layers[idx + 1] if idx + 1 < len(layers) else None
    if bn_layer is not None and isinstance(bn_layer, tf.keras.layers.BatchNormalization):
        gamma = bn_layer.gamma.numpy()
        beta = bn_layer.beta.numpy()
        mean = bn_layer.moving_mean.numpy()
        var = bn_layer.moving_variance.numpy()
        eps = bn_layer.epsilon

        alpha = gamma / np.sqrt(var + eps)
        alpha = np.where(np.abs(alpha) < 1e-6, np.sign(alpha) * 1e-6 + 1e-6, alpha)
        z_threshold = mean - beta / alpha

        sense = np.where(alpha >= 0, 1, -1).astype(np.int8)
        popcount_threshold = (z_threshold + metadata["input_dim"] - bias) / 2.0

        artifacts.extend(
            [
                (f"{layer.name}_bn_alpha.npy", alpha.astype(np.float32)),
                (f"{layer.name}_bn_mean.npy", mean.astype(np.float32)),
                (f"{layer.name}_bn_var.npy", var.astype(np.float32)),
                (f"{layer.name}_bn_beta.npy", beta.astype(np.float32)),
                (f"{layer.name}_popcount_threshold.npy", popcount_threshold.astype(np.float32)),
                (f"{layer.name}_comparison_sense.npy", sense),
            ]
        )

        metadata.update(
            {
                "bn_alpha_file": f"{layer.name}_bn_alpha.npy",
                "bn_mean_file": f"{layer.name}_bn_mean.npy",
                "bn_var_file": f"{layer.name}_bn_var.npy",
                "bn_beta_file": f"{layer.name}_bn_beta.npy",
                "popcount_threshold_file": f"{layer.name}_popcount_threshold.npy",
                "comparison_sense_file": f"{layer.name}_comparison_sense.npy",
            }
        )
        threshold_q = quantize_to_int16(popcount_threshold)
        sense_bits = (sense > 0).astype(np.uint8)
    else:
        threshold_q = None
        sense_bits = np.ones((layer.units,), dtype=np.uint8)

    memh_blobs.append(
        {
            "name": layer.name,
            "weights_bits": weights_bits.copy(),
            "bias_q": quantize_to_int16(bias),
            "threshold_q": threshold_q,
            "sense_bits": sense_bits,
        }
    )

    for filename, array in artifacts:
        np.save(ARTIFACT_DIR / filename, array)

    manifest.append(metadata)

manifest_path = ARTIFACT_DIR / "binary_layer_manifest.json"
with manifest_path.open("w", encoding="utf-8") as fp:
    json.dump(manifest, fp, indent=2)

print(f"Exported BinaryDense layers -> {manifest_path}")
# Build flattened memory in hardware layout:
# L1 weights -> L1 bias -> L1 thresholds -> L2 weights -> L2 bias
L1_IN = images.shape[1]
L1_OUT = HIDDEN_UNITS
L2_OUT = len(target_names)
L2_IN = L1_OUT

L1_W_BITS = L1_IN * L1_OUT
L1_W_WORDS = (L1_W_BITS + 31) // 32
L1_B_WORDS = L1_OUT
L1_T_WORDS = L1_OUT
L1_S_WORDS = (L1_OUT + 31) // 32
L2_W_BITS = L2_IN * L2_OUT
L2_W_WORDS = (L2_W_BITS + 31) // 32
L2_B_WORDS = L2_OUT
MEM_DEPTH = L1_W_WORDS + L1_B_WORDS + L1_T_WORDS + L1_S_WORDS + L2_W_WORDS + L2_B_WORDS

mem_words: List[int] = []

# Layer ordering matches memh_blobs insertion (hidden, then logits)
hidden_blob, logits_blob = memh_blobs

mem_words.extend(pack_bits_to_words(hidden_blob["weights_bits"]))
mem_words.extend(int(x) & 0xFFFF for x in hidden_blob["bias_q"].flatten())
hidden_thr = hidden_blob["threshold_q"] if hidden_blob["threshold_q"] is not None else np.zeros((L1_OUT,), dtype=np.int16)
mem_words.extend(int(x) & 0xFFFF for x in hidden_thr.flatten())
mem_words.extend(pack_bits_to_words(hidden_blob["sense_bits"]))

mem_words.extend(pack_bits_to_words(logits_blob["weights_bits"]))
mem_words.extend(int(x) & 0xFFFF for x in logits_blob["bias_q"].flatten())

write_svh(mem_words, MEM_DEPTH, HW_SVH_PATH)


# %%
# Golden vectors -------------------------------------------------------------
# Hardware-model prediction helper using quantized blobs.
def predict_hw(images_bits: np.ndarray) -> np.ndarray:
    hidden_blob, logits_blob = memh_blobs
    w1 = hidden_blob["weights_bits"]
    thr = hidden_blob["threshold_q"]
    sense = hidden_blob["sense_bits"]
    w2 = logits_blob["weights_bits"]
    bias2 = logits_blob["bias_q"]

    preds: List[int] = []
    for img in images_bits.reshape(images_bits.shape[0], -1):
        hidden = np.zeros((L1_OUT,), dtype=np.uint8)
        for o in range(L1_OUT):
            match_count = int(np.sum(img == w1[:, o]))
            if sense[o] > 0:
                hidden[o] = 1 if match_count >= thr[o] else 0
            else:
                hidden[o] = 1 if match_count <= thr[o] else 0

        best_cls = 0
        best_score = -1_000_000
        for c in range(L2_OUT):
            match_count = int(np.sum(hidden == w2[:, c]))
            score = (match_count * 2 - L2_IN) + int(bias2[c])
            if score > best_score:
                best_score = score
                best_cls = c
        preds.append(best_cls)
    return np.array(preds, dtype=np.int32)


golden_count = min(NUM_GOLDEN, len(X_test))
golden_images = X_test[:golden_count]
golden_labels = y_test[:golden_count]
golden_probs = test_probs[:golden_count]
golden_preds = np.argmax(golden_probs, axis=1)
golden_preds_hw = predict_hw(((golden_images >= 0).astype(np.uint8)))

golden_payload = {
    "images_flat": golden_images.astype(np.float32),
    "images_8x8": golden_images.reshape(golden_count, 8, 8),
    "images_bits": ((golden_images >= 0).astype(np.uint8)),
    "labels": golden_labels.astype(np.int32),
    "predictions": golden_preds.astype(np.int32),
    "predictions_hw": golden_preds_hw.astype(np.int32),
    "probabilities": golden_probs.astype(np.float32),
}

np.savez(ARTIFACT_DIR / "golden_vectors.npz", **golden_payload)
print(f"Stored {golden_count} golden samples at {ARTIFACT_DIR / 'golden_vectors.npz'}")


# %%
# Model persistence ----------------------------------------------------------
model_save_path = ARTIFACT_DIR / "digits_bnn.keras"
model.save(model_save_path)
print(f"Saved trained model to {model_save_path}")
