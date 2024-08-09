#train.py
import os
import tensorflow as tf
from callbacks import CallbackEval
from model_builder import build_model
from data_loader import load_and_preprocess_data, prepare_datasets

# Configurations
data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
batch_size = 32
epochs = 100

# Load and preprocess the data
train_data, validation_data, wavs_path = load_and_preprocess_data(data_url)

# Define the characters
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_int = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
int_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_int.get_vocabulary(), oov_token="", invert=True)

# Prepare datasets
train_dataset, validation_dataset = prepare_datasets(train_data, validation_data, batch_size, wavs_path, char_to_int)

# Build the model
model = build_model(input_dim=384 // 2 + 1, output_dim=char_to_int.vocabulary_size(), rnn_units=512)
model.summary(line_length=110)

# Directories for saving checkpoints and the final model
checkpoint_dir = 'models/model_mark_1/checkpoint'
final_model_dir = 'models/model_mark_1/models'
graph_dir = 'models/model_mark_1/graphs'

# Ensure directories exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

# Callbacks
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(checkpoint_dir, 'model_epoch_{epoch:03d}.ckpt'),
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)
validation_callback = CallbackEval(validation_dataset, int_to_char, model)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback, model_checkpoint_callback],
)

# Save the final model in .keras format
model.save(os.path.join(final_model_dir, 'model_mark_1'))

# Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the loss graph
plt.savefig(os.path.join(graph_dir, 'loss_graph.png'))
plt.close()