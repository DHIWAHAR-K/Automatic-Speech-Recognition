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

To run the Flask application, navigate to the project directory in your terminal and execute:

```bash
python app.py
```

This command will start the Flask server, making the API available for movie recommendations. Interact with the API by sending a GET request to /recommend with a movie title as a query parameter, like so:

```bash
http://127.0.0.1:5000/recommend?title=The%20Dark%20Knight%20Rises
```

## Features

1. Data Loading and Parsing: Automated scripts to load and parse the TMDB 5000 movie dataset.

2. Content-Based Recommendation: Utilizes TF-IDF vectorization and cosine similarity to find movies similar to a given title.

3. Flask API: A simple and efficient web API to interact with the recommendation system.


## License

This README accurately reflects the Flask setup, dependencies, and usage, replacing Streamlit references with Flask and adjusting the paths and method descriptions accordingly. Feel free to copy and paste this content directly into your `README.md` file.