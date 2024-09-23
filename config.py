#config.py
import tensorflow as tf
from tensorflow import keras

class Config:
    # Dataset URL and paths
    DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    DATA_PATH = keras.utils.get_file("LJSpeech-1.1", DATA_URL, untar=True)
    WAVS_PATH = DATA_PATH + "/wavs/"
    METADATA_PATH = DATA_PATH + "/metadata.csv"
    
    # Directories for saving checkpoints, graphs, and model
    CHECKPOINT_DIR = "checkpoints"
    GRAPHS_DIR = "graphs"
    MODEL_DIR = "model"
    
    # Data processing settings
    FRAME_LENGTH = 256  
    FRAME_STEP = 160    
    FFT_LENGTH = 384    
    
    # Training/Validation split ratio and batch size
    TRAIN_SPLIT = 0.90
    BATCH_SIZE = 16
    
    # Model settings
    RNN_UNITS = 512     
    EPOCHS = 100        
    
    # Set of characters for transcription
    CHARACTERS = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    
    # Vocabulary mappings
    CHAR_TO_NUM = tf.keras.layers.StringLookup(vocabulary=CHARACTERS, oov_token="")
    NUM_TO_CHAR = tf.keras.layers.StringLookup(vocabulary=CHAR_TO_NUM.get_vocabulary(), invert=True)
    
    # Vocabulary size
    VOCAB_SIZE = len(CHAR_TO_NUM.get_vocabulary())

    @staticmethod
    def decode_batch_predictions(pred):
        """Decode predictions from the model output to readable text."""
        input_len = tf.ones(pred.shape[0]) * pred.shape[1]
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(Config.NUM_TO_CHAR(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text