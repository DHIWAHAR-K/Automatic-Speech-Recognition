#prepare_data.py
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def download_and_prepare_data(data_url, split_ratio=0.90):
    data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
    wavs_path = data_path + "/wavs/"
    metadata_path = data_path + "/metadata.csv"

    # Read metadata file and parse it
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["ID", "Transcription", "Normalized Transcription"]
    metadata_df = metadata_df[["ID", "Normalized Transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    split = int(len(metadata_df) * split_ratio)
    train_data = metadata_df[:split]
    validation_data = metadata_df[split:]

    return train_data, validation_data, wavs_path
