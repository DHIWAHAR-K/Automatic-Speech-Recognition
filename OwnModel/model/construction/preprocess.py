import torch
import os
import torchaudio
from glob import glob
from audio_processing import load_audio, pad_or_trim

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def load_audio(self, file_path):
        waveform = load_audio(file_path, sr=self.sample_rate)
        return waveform
    
    def preprocess_audio(self, file_path):
        waveform = self.load_audio(file_path)
        waveform = pad_or_trim(waveform)
        mel_spec = self.mel_spectrogram(torch.tensor(waveform).unsqueeze(0))
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db

    def preprocess_dataset(self, dataset_path):
        audio_files = glob(os.path.join(dataset_path, '**/*.wav'), recursive=True)
        processed_audios = []
        for file in audio_files:
            mel_spec_db = self.preprocess_audio(file)
            processed_audios.append((file, mel_spec_db))
        return processed_audios