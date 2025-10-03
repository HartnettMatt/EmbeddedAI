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

# %%
# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# %%
# Download data

DATASET_PATH = (
    pathlib.Path(__file__).parent.resolve() / "../testdata/mini_speech_commands/"
)

if not DATASET_PATH.exists():
    os.mkdir(DATASET_PATH)
if not any(DATASET_PATH.iterdir()):
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=".",
        cache_subdir=DATASET_PATH,
    )
else:
    print(f"Files found at {DATASET_PATH}, not downloading new files")

# %%
