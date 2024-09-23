#train.py
import os
from config import Config
from tensorflow import keras
from data_loader import load_data
from callbacks import get_callbacks
from model_builder import build_model

# Ensure necessary directories exist
if not os.path.exists(Config.CHECKPOINT_DIR):
    os.makedirs(Config.CHECKPOINT_DIR)
if not os.path.exists(Config.GRAPHS_DIR):
    os.makedirs(Config.GRAPHS_DIR)
if not os.path.exists(Config.MODEL_DIR):
    os.makedirs(Config.MODEL_DIR)

# Load datasets
train_dataset, validation_dataset = load_data()

# Build model
model = build_model(input_dim=Config.FFT_LENGTH // 2 + 1, output_dim=Config.VOCAB_SIZE, rnn_units=Config.RNN_UNITS)

# Train model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=Config.EPOCHS,
    callbacks=get_callbacks(validation_dataset),
)

# Save the final model
model.save(os.path.join(Config.MODEL_DIR, 'final_model.h5'))
print(f"Trained model saved in the '{Config.MODEL_DIR}' folder.")