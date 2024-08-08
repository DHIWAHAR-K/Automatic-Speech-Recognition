#config.py
import tensorflow as tf

# Data processing parameters
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384
BATCH_SIZE = 32

# Characters and mappings
CHARACTERS = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
CHAR_TO_INT = tf.keras.layers.StringLookup(vocabulary=CHARACTERS, oov_token="")
INT_TO_CHAR = tf.keras.layers.StringLookup(vocabulary=CHAR_TO_INT.get_vocabulary(), oov_token="", invert=True)

# Directory paths
DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
CHECKPOINT_DIR = 'checkpoints'
FINAL_MODEL_DIR = 'models'
GRAPH_DIR = 'graphs'

# Training parameters
EPOCHS = 100
RNN_LAYERS = 5
RNN_UNITS = 128
