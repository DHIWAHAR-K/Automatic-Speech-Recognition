import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import SpeechRecognition
from preprocess import AudioPreprocessor
from text_process import GreedyDecoder
from error_metrics import wer, cer, avg_wer
import os

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, mel_spec_db = self.data[idx]
        # Dummy label and length for simplicity
        label = torch.randint(0, 29, (10,))
        label_length = torch.tensor([10])
        return mel_spec_db, label, label_length

def collate_fn(batch):
    mel_specs, labels, label_lengths = zip(*batch)
    mel_specs = torch.nn.utils.rnn.pad_sequence(mel_specs, batch_first=True)
    labels = torch.stack(labels)
    label_lengths = torch.stack(label_lengths)
    return mel_specs, labels, label_lengths

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_wer = 0
    total_cer = 0
    combined_ref_len = 0

    for inputs, labels, label_lengths in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        hidden = model._init_hidden(inputs.size(0))
        optimizer.zero_grad()
        outputs, _ = model(inputs, hidden)
        outputs = outputs.transpose(0, 1)  # Time-first for loss calculation
        loss = criterion(outputs.log_softmax(2), labels, torch.tensor([outputs.size(0)] * outputs.size(1)), label_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Decode and calculate WER and CER
        decodes, targets = GreedyDecoder(outputs, labels, label_lengths)
        for decode, target in zip(decodes, targets):
            total_wer += wer(target, decode)
            total_cer += cer(target, decode)
            combined_ref_len += len(target.split())

    avg_wer_score = avg_wer([total_wer], combined_ref_len)
    avg_cer_score = avg_wer([total_cer], combined_ref_len)
    print(f"Average WER: {avg_wer_score:.4f}, Average CER: {avg_cer_score:.4f}")

    return total_loss / len(train_loader)

def main():
    dataset_path = "/path/to/MLCommons/dataset"
    preprocessor = AudioPreprocessor()
    processed_data = preprocessor.preprocess_dataset(dataset_path)
    train_dataset = AudioDataset(processed_data)
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeechRecognition(hidden_size=1024, num_classes=29, n_feats=81, num_layers=1, dropout=0.1)
    model.to(device)
    criterion = torch.nn.CTCLoss(blank=28)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range (10):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()