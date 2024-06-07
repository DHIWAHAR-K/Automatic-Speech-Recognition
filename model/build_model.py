#build_model.py
import tensorflow as tf
from tensorflow import keras
from utils.loss import CTCLoss

def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    input_spectrogram = tf.keras.layers.Input((None, input_dim), name="input")
    x = tf.keras.layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False, name="conv_1")(x)
    x = tf.keras.layers.BatchNormalization(name="conv_1_bn")(x)
    x = tf.keras.layers.ReLU(name="conv_1_relu")(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False, name="conv_2")(x)
    x = tf.keras.layers.BatchNormalization(name="conv_2_bn")(x)
    x = tf.keras.layers.ReLU(name="conv_2_relu")(x)
    x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    
    for i in range(1, rnn_layers + 1):
        recurrent = tf.keras.layers.LSTM(units=rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, name=f"lstm_{i}")
        x = tf.keras.layers.Bidirectional(recurrent, name=f"bidirectional_{i}", merge_mode="concat")(x)
        if i < rnn_layers:
            x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    x = tf.keras.layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = tf.keras.layers.ReLU(name="dense_1_relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    output = tf.keras.layers.Dense(units=output_dim + 1, activation="softmax")(x)
    
    model = keras.Model(input_spectrogram, output)
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss=CTCLoss)
    
    return model
