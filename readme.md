# Speech-to-Text Model with CNN + BiLSTM + CTC

This project implements a speech-to-text model using a Convolutional Neural Network (CNN) combined with Bidirectional Long Short-Term Memory (BiLSTM) layers and the Connectionist Temporal Classification (CTC) loss function. The model is trained on the LJSpeech dataset and supports modular components for data loading, model building, and training.

## Project Structure

- `train.py`: The main script that initializes the model, loads data, and trains the model while saving checkpoints and performance graphs.
- `data_loader.py`: Handles loading, preprocessing, and dataset preparation from the LJSpeech dataset.
- `model_builder.py`: Builds the CNN + BiLSTM + CTC model and compiles it with the custom CTC loss function.
- `callbacks.py`: Implements a custom callback for evaluation, calculating the Word Error Rate (WER) after each epoch.
- `config.py`: Contains configuration parameters for easy modification of hyperparameters and dataset paths.

## Setup and Installation

Ensure you have Python 3.6+ installed and TensorFlow configured, then proceed with the following steps to set up the project environment:

1. Install required Python packages:

    ```bash
    pip install tensorflow pandas matplotlib jiwer
    ```

2. Download the LJSpeech dataset and ensure it is accessible by the `data_loader.py` script:

    - Dataset URL: `https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2`

## Usage

To train the model, navigate to the project directory in your terminal and execute:

```bash
python train.py
```

This command will initiate the data loading, model building, and training process. The model checkpoints will be saved in the checkpoints directory. After training, the final model will be saved in the model directory, and a loss graph will be generated in the graphs directory.

## Usage