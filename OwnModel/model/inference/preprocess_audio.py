import torch
from audio_processing import load_audio, pad_or_trim, log_mel_spectrogram

def preprocess_audio(file_path, device):
    waveform = load_audio(file_path)
    waveform = pad_or_trim(waveform)
    mel_spec = log_mel_spectrogram(waveform)
    return mel_spec.unsqueeze(0).to(device)  # Add batch dimension and move to device