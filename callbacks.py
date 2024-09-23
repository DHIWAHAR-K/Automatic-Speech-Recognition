#callbacks.py
import os
import numpy as np
from jiwer import wer
from config import Config
from tensorflow import keras
import matplotlib.pyplot as plt

def get_callbacks(validation_dataset):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(Config.CHECKPOINT_DIR, 'epoch_{epoch:02d}.h5'),
        save_weights_only=False,
        verbose=1
    )

    class CallbackEval(keras.callbacks.Callback):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def on_epoch_end(self, epoch, logs=None):
            predictions = []
            targets = []
            for batch in self.dataset:
                X, y = batch
                batch_predictions = self.model.predict(X)
                batch_predictions = Config.decode_batch_predictions(batch_predictions)
                predictions.extend(batch_predictions)
                for label in y:
                    label = tf.strings.reduce_join(Config.NUM_TO_CHAR(label)).numpy().decode("utf-8")
                    targets.append(label)
            wer_score = wer(targets, predictions)
            print(f"Word Error Rate: {wer_score:.4f}")

    class LossPlotCallback(keras.callbacks.Callback):
        def __init__(self):
            self.epochs = []
            self.train_losses = []
            self.val_losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.epochs.append(epoch + 1)
            self.train_losses.append(logs['loss'])
            self.val_losses.append(logs.get('val_loss', 0))

            plt.figure()
            plt.plot(self.epochs, self.train_losses, label="Training Loss")
            plt.plot(self.epochs, self.val_losses, label="Validation Loss")
            plt.title("Epoch vs Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{Config.GRAPHS_DIR}/epoch_vs_loss_epoch_{epoch + 1}.png")
            plt.close()

    return [checkpoint_callback, CallbackEval(validation_dataset), LossPlotCallback()]