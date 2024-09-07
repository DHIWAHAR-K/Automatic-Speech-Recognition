# Speech-to-Text Model with CNN + BiLSTM + CTC

This project implements a speech-to-text model using a Convolutional Neural Network (CNN) combined with Bidirectional Long Short-Term Memory (BiLSTM) layers and the Connectionist Temporal Classification (CTC) loss function. The model is trained on the LJSpeech dataset and supports modular components for data loading, model building, and training.

## Project Structure

- `train.py`' The main script that initializes the model, loads data, and trains the model while saving checkpoints and performance graphs.
- `data_loader.py`: Handles loading, preprocessing, and dataset preparation from the LJSpeech dataset.
- `model_builder.py`: Builds the CNN + BiLSTM + CTC model and compiles it with the custom CTC loss function.
- `callbacks.py`: Implements a custom callback for evaluation, calculating the Word Error Rate (WER) after each epoch.
- `config.py`: Contains configuration parameters for easy modification of hyperparameters and dataset paths.

## Setup and Installation

Ensure you have Python 3.6+ installed and proceed with the following steps to set up the project environment:

Install the following Python packages:

    ```bash
    pip install tensorflow pandas matplotlib jiwer
    ```

## Usage

To train the model, navigate to the project directory in your terminal and execute:

```bash
python train.py
```

This command will initiate the data loading, model building, and training process. The model checkpoints will be saved in the models/checkpoint directory. After training, the final model will be saved in the models/model_mark_1/models directory, and a loss graph will be generated in the models/graphs directory.

## Features

1.	Data Loading and Preprocessing: The LJSpeech dataset is automatically downloaded and preprocessed, including audio file transformation and text normalization.
2.	CNN + BiLSTM + CTC Model: A custom-built model that combines convolutional layers with BiLSTM and uses the CTC loss function to handle variable-length sequences.
3.	Word Error Rate (WER) Calculation: A custom callback evaluates the model’s performance by calculating the WER on the validation dataset at the end of each epoch.
4.	Checkpointing and Model Saving: Checkpoints are saved after every epoch, and the final trained model is stored in .keras format for future inference.
5.	Training Loss Plot: A graph of training and validation loss is generated to visually track the model’s performance over time.


## License

Feel free to copy and paste this README structure into your README.md file. This README accurately reflects the setup, dependencies, and usage for the speech-to-text model.