#data_preprocessing.py
import os
import pandas as pd
import tensorflow as tf

def load_metadata(metadata_path):
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["ID", "Transcription", "Normalized Transcription"]
    metadata_df = metadata_df[["ID", "Normalized Transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    return metadata_df

def split_data(metadata_df, split_ratio=0.90):
    split = int(len(metadata_df) * split_ratio)
    train_data = metadata_df[:split]
    validation_data = metadata_df[split:]
    return train_data, validation_data

def encode_single_sample(wavs_path, char_to_int, frame_length, frame_step, fft_length, wav_file, label):
    file = tf.io.read_file(os.path.join(wavs_path, wav_file + ".wav"))
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_int(label)
    return spectrogram, label

def prepare_datasets(train_data, validation_data, wavs_path, char_to_int, frame_length, frame_step, fft_length, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((list(train_data["ID"]), list(train_data["Normalized Transcription"])))
    train_dataset = train_dataset.map(lambda x, y: encode_single_sample(wavs_path, char_to_int, frame_length, frame_step, fft_length, x, y), num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    validation_dataset = tf.data.Dataset.from_tensor_slices((list(validation_data["ID"]), list(validation_data["Normalized Transcription"])))
    validation_dataset = validation_dataset.map(lambda x, y: encode_single_sample(wavs_path, char_to_int, frame_length, frame_step, fft_length, x, y), num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, validation_dataset