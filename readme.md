# Automatic Speech Recognition

End-to-end speech-to-text model inspired by DeepSpeech 2. Combines CNN feature extraction with stacked Bidirectional LSTM layers and CTC loss for sequence-to-sequence audio transcription. Trained on the LJSpeech dataset.

## Model Architecture

```
Audio Waveform
    │
    ▼
┌────────────────────┐
│  STFT Spectrogram   │  Frame length: 256, Frame step: 160, FFT: 384
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Conv2D Block 1    │  32 filters, kernel [11, 41], stride [2, 2] + BatchNorm + ReLU
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Conv2D Block 2    │  32 filters, kernel [11, 21], stride [1, 2] + BatchNorm + ReLU
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  5× Bidirectional   │  512 units each, concat merge mode, 50% dropout between layers
│      LSTM           │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Dense + ReLU       │  1024 units + 50% dropout
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Softmax Output     │  Vocabulary size + 1 (CTC blank token)
└────────────────────┘

Loss: CTC (Connectionist Temporal Classification)
Optimizer: Adam (lr=1e-4)
```

## Features

- **DeepSpeech 2-inspired architecture** — CNN front-end for local feature extraction + BiLSTM for temporal modeling
- **CTC loss function** — Handles variable-length alignment between audio frames and text without forced alignment
- **Word Error Rate tracking** — Custom callback evaluates WER on validation set after each epoch
- **Training visualization** — Automatic loss plots saved to `graphs/` directory after each epoch
- **Model checkpointing** — Full model saved at every epoch to `checkpoints/`
- **Greedy CTC decoding** — Beam-free decoding for fast inference

## Project Structure

```
Automatic-Speech-Recognition/
├── config.py           # Hyperparameters, paths, character vocabulary, CTC decode utility
├── data_loader.py      # LJSpeech loader: WAV → STFT spectrogram + character-level labels
├── model_builder.py    # DeepSpeech 2 model: 2× Conv2D + 5× BiLSTM + Dense + CTC loss
├── train.py            # Training script with dataset loading, model building, and saving
├── callbacks.py        # Custom callbacks: WER evaluation, loss plotting, checkpointing
├── requirements.txt    # Dependencies
└── notebook/
    └── model.ipynb     # Jupyter notebook for experimentation
```

## Dataset

**LJSpeech-1.1** — 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. ~24 hours of speech.

- Automatically downloaded via `keras.utils.get_file()`
- 90/10 train/validation split
- Audio preprocessed to STFT spectrograms with power-law compression and z-normalization
- Text labels mapped to character-level indices (a-z, punctuation, space)

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FRAME_LENGTH` | 256 | STFT window size |
| `FRAME_STEP` | 160 | STFT hop length |
| `FFT_LENGTH` | 384 | FFT size |
| `RNN_UNITS` | 512 | BiLSTM hidden units |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 100 | Training epochs |
| `TRAIN_SPLIT` | 0.90 | Train/val ratio |

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py
```

The model will be saved to `model/final_model.h5` after training.

## Tech Stack

**Framework:** TensorFlow 2.x / Keras

**Audio Processing:** `tf.signal` (STFT, spectrogram)

**Evaluation:** `jiwer` (Word Error Rate)

**Visualization:** Matplotlib
