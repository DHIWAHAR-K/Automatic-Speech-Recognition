#preprocessing.py
import tensorflow as tf
from tensorflow import keras

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_int = keras.layers.StringLookup(vocabulary=characters, oov_token="")
int_to_char = keras.layers.StringLookup(vocabulary=char_to_int.get_vocabulary(), oov_token="", invert=True)

def encode_single_sample(wavs_path, wav_file, label):
    frame_length = 256
    frame_step = 160
    fft_length = 384

    file = tf.io.read_file(wavs_path + wav_file + ".wav")
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

def create_datasets(train_data, validation_data, wavs_path, batch_size=32):
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(train_data["ID"]), list(train_data["Normalized Transcription"]))
    )
    train_dataset = (
        train_dataset.map(lambda x, y: encode_single_sample(wavs_path, x, y), num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(validation_data["ID"]), list(validation_data["Normalized Transcription"]))
    )
    validation_dataset = (
        validation_dataset.map(lambda x, y: encode_single_sample(wavs_path, x, y), num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    return train_dataset, validation_dataset
