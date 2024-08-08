#train.py
import os
import numpy as np
from jiwer import wer
import tensorflow as tf
from tensorflow import keras

class CallbackEval(keras.callbacks.Callback):
    def __init__(self, dataset, model, int_to_char):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.int_to_char = int_to_char

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(self.int_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = self.decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = tf.strings.reduce_join(self.int_to_char(label)).numpy().decode("utf-8")
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)

def train_model(model, train_dataset, validation_dataset, epochs, checkpoint_dir, graph_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'model_epoch_{epoch:03d}.ckpt'),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    validation_callback = CallbackEval(validation_dataset, model, int_to_char)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback, model_checkpoint_callback]
    )

    return history