#data_loader.py
import pandas as pd
import tensorflow as tf
from config import Config

def encode_single_sample(wav_file, label):
    file = tf.io.read_file(Config.WAVS_PATH + wav_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(
        audio, frame_length=Config.FRAME_LENGTH, frame_step=Config.FRAME_STEP, fft_length=Config.FFT_LENGTH
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = Config.CHAR_TO_NUM(label)
    return spectrogram, label

def load_data():
    metadata_df = pd.read_csv(Config.METADATA_PATH, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    
    split = int(len(metadata_df) * Config.TRAIN_SPLIT)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(Config.BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
    )
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(Config.BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset