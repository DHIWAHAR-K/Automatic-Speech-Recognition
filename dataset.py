#dataset.py
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import TextProcess

class SpecAugment(nn.Module):
    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        self.rate = rate  # Rate of augmentation
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),  # Frequency masking
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)  # Time masking
        )
        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),  # Frequency masking
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),  # Time masking
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),  # Frequency masking
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)  # Time masking
        )
        policies = {1: self.policy1, 2: self.policy2, 3: self.policy3}
        self._forward = policies[policy]  # Select augmentation policy

    def forward(self, x):
        return self._forward(x)  # Apply selected policy

    def policy1(self, x):
        probability = torch.rand(1, 1).item()  # Random probability
        if self.rate > probability:
            return self.specaug(x)  # Apply specaug
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()  # Random probability
        if self.rate > probability:
            return self.specaug2(x)  # Apply specaug2
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()  # Random probability
        if probability > 0.5:
            return self.policy1(x)  # Apply policy1
        return self.policy2(x)  # Apply policy2

class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length
        )

    def forward(self, x):
        x = self.transform(x)  # Compute mel spectrogram
        x = torch.log1p(x)  # Apply logarithmic transformation
        return x

def get_featurizer(sample_rate, n_feats=81):
    return LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)

class Data(torch.utils.data.Dataset):
    parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15
    }

    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, specaug_policy,
                 time_mask, freq_mask, valid=False, shuffle=True, text_to_int=True, log_ex=True):
        self.log_ex = log_ex  # Log exceptions flag
        self.text_process = TextProcess()  # Text processing
        self.data = pd.read_json(json_path, lines=True)  # Load data from JSON file

        if valid:
            self.audio_transforms = nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )

    def __len__(self):
        return len(self.data)  # Return length of dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()  # Convert tensor to int

        while True:
            try:
                file_path = self.data.key.iloc[idx]  # Get file path
                waveform, _ = torchaudio.load(file_path)  # Load audio file
                label = self.text_process.text_to_int_sequence(self.data['text'].iloc[idx])  # Convert text to integer sequence
                spectrogram = self.audio_transforms(waveform)  # Apply audio transformations
                spec_len = spectrogram.shape[-1] // 2  # Spectrogram length
                label_len = len(label)  # Label length
                if spec_len < label_len:
                    raise ValueError('Spectrogram length is smaller than label length')
                if spectrogram.shape[0] > 1:
                    raise ValueError('Dual channel, skipping audio file %s' % file_path)
                if spectrogram.shape[2] > 1650:
                    raise ValueError('Spectrogram too big, size %s' % spectrogram.shape[2])
                if label_len == 0:
                    raise ValueError('Label length is zero, skipping %s' % file_path)
                break
            except Exception as e:
                if self.log_ex:
                    print(str(e), file_path)
                idx = (idx - 1) if idx != 0 else (idx + 1)

        return spectrogram, label, spec_len, label_len  # Return processed data

    def describe(self):
        return self.data.describe()  # Describe dataset statistics

def collate_fn_padd(data):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))  # Transpose spectrogram
        labels.append(torch.Tensor(label))  # Convert label to tensor
        input_lengths.append(input_length)  # Append input length
        label_lengths.append(label_length)  # Append label length

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths  # Return padded sequences