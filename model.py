#model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)  # Apply dropout to prevent overfitting
        self.norm = nn.LayerNorm(n_feats)  # Layer normalization
        self.keep_shape = keep_shape  # Flag to keep the input shape unchanged

    def forward(self, x):
        x = x.transpose(1, 2)  # Transpose for LayerNorm
        x = self.dropout(F.gelu(self.norm(x)))  # Apply LayerNorm, activation, and dropout
        if self.keep_shape:
            return x.transpose(1, 2)  # Transpose back if needed
        else:
            return x

class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29,
        "n_feats": 81,
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1
    }

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout, bidirectional=False):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Define CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10 // 2),
            ActDropNormCNN1D(n_feats, dropout),
        )

        # Define dense layers
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.0,
                            bidirectional=bidirectional)

        # Define normalization and dropout
        self.layer_norm2 = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

        self._init_weights()  # Initialize weights

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        num_directions = 2 if self.bidirectional else 1
        return (torch.zeros(n * num_directions, batch_size, hs),
                torch.zeros(n * num_directions, batch_size, hs))

    def forward(self, x, hidden):
        x = x.squeeze(1)  # Remove channel dimension
        x = self.cnn(x)  # Apply CNN layer
        x = self.dense(x)  # Apply dense layers
        x = x.transpose(0, 1)  # Transpose for LSTM
        out, (hn, cn) = self.lstm(x, hidden)  # Apply LSTM
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # Apply normalization and dropout
        return self.final_fc(x), (hn, cn)  # Final output