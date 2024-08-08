#main.py
import os
import model as md
import train as tr
import utils as ut
import config as cfg
import tensorflow as tf
from tensorflow import keras
import data_processing as dp

def main():
    # Download and prepare data
    data_path = keras.utils.get_file("LJSpeech-1.1", cfg.DATA_URL, untar=True)
    wavs_path = os.path.join(data_path, "wavs")
    metadata_path = os.path.join(data_path, "metadata.csv")

    metadata_df = dp.load_metadata(metadata_path)
    train_data, validation_data = dp.split_data(metadata_df)

    print(f"Size of the training set: {len(train_data)}")
    print(f"Size of the validation set: {len(validation_data)}")

    train_dataset, validation_dataset = dp.prepare_datasets(
        train_data, validation_data, wavs_path, cfg.CHAR_TO_INT,
        cfg.FRAME_LENGTH, cfg.FRAME_STEP, cfg.FFT_LENGTH, cfg.BATCH_SIZE
    )

    model = md.build_model(
        input_dim=cfg.FFT_LENGTH // 2 + 1,
        output_dim=cfg.CHAR_TO_INT.vocabulary_size(),
        rnn_units=cfg.RNN_UNITS
    )
    model.summary(line_length=110)

    history = tr.train_model(model, train_dataset, validation_dataset, cfg.EPOCHS, cfg.CHECKPOINT_DIR, cfg.GRAPH_DIR)

    model.save(os.path.join(cfg.FINAL_MODEL_DIR, 'model_mark_1'))
    model.save(os.path.join(cfg.FINAL_MODEL_DIR, 'model_mark_1.h5'))
    model.save(os.path.join(cfg.FINAL_MODEL_DIR, 'model_mark_1.keras'))

    ut.plot_and_save_history(history, cfg.GRAPH_DIR)

if __name__ == "__main__":
    main()