#app.py
from data.prepare_data import download_and_prepare_data
from utils.preprocessing import create_datasets, char_to_int
from model.train_model import train_model

data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

# Prepare data
train_data, validation_data, wavs_path = download_and_prepare_data(data_url)

# Create datasets
train_dataset, validation_dataset = create_datasets(train_data, validation_data, wavs_path)

# Train the model
train_model(
    train_dataset,
    validation_dataset,
    input_dim=384 // 2 + 1,
    output_dim=char_to_int.vocabulary_size(),
    epochs=100,
    rnn_units=512
)