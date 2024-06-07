#train_model.py
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utils.callbacks import CallbackEval
from model.build_model import build_model
from utils.preprocessing import create_datasets

def train_model(train_dataset, validation_dataset, input_dim, output_dim, epochs=100, rnn_units=512):
    # Build the model
    model = build_model(input_dim=input_dim, output_dim=output_dim, rnn_units=rnn_units)
    model.summary(line_length=110)

    # Define directories for saving checkpoints and the final model
    checkpoint_dir = '../models/model_mark_1/checkpoint'
    final_model_dir = '../models/model_mark_1/models'
    graph_dir = '../models/model_mark_1/graphs'

    # Ensure directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # ModelCheckpoint callback to save the model checkpoint after each epoch
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'model_epoch_{epoch:03d}.ckpt'),
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    # EarlyStopping callback to stop training when the validation loss stops improving
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    # Callback function to check transcription on the validation set.
    validation_callback = CallbackEval(validation_dataset, model)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback, model_checkpoint_callback, early_stopping_callback],
    )

    # Save the final model in .keras format
    final_model_path_1 = os.path.join(final_model_dir, 'model_mark_1')
    model.save(final_model_path_1)
    final_model_path_2 = os.path.join(final_model_dir, 'model_mark_1.h5')
    model.save(final_model_path_2)
    final_model_path_3 = os.path.join(final_model_dir, 'model_mark_1.keras')
    model.save(final_model_path_3)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the loss graph
    graph_path = os.path.join(graph_dir, 'loss_graph.png')
    plt.savefig(graph_path)
    plt.close()
