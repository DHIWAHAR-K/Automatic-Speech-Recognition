#train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import SpeechRecognition
from torch.utils.data import DataLoader
from utils import wer, cer, GreedyDecoder
from dataset import Data, collate_fn_padd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 64

class SpeechModule(LightningModule):
    def __init__(self, model):
        super(SpeechModule, self).__init__()
        self.model = model  # Speech recognition model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)  # CTC loss function
        self.train_losses = []  # Store training losses for plotting
        self.val_losses = []  # Store validation losses for plotting

    def forward(self, x, hidden):
        return self.model(x, hidden)  # Forward pass

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), LEARNING_RATE)  # Optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.50, patience=6  # Learning rate scheduler
        )
        return [self.optimizer], [self.scheduler]

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch  # Unpack batch
        bs = spectrograms.shape[0]  # Batch size
        hidden = self.model._init_hidden(bs)  # Initialize hidden state
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)  # Move to device
        output, _ = self(spectrograms, (hn, c0))  # Forward pass
        output = nn.functional.log_softmax(output, dim=2)  # Log softmax activation
        loss = self.criterion(output, labels, input_lengths, label_lengths)  # Compute loss
        return loss, output, labels, label_lengths

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)  # Compute loss for training step
        self.train_losses.append(loss.item())  # Append training loss for plotting
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # Log training loss
        return {'loss': loss}

    def train_dataloader(self):
        d_params = Data.parameters  # Data parameters
        train_dataset = Data(json_path='data/train.json', **d_params)  # Training dataset
        return DataLoader(
            dataset=train_dataset, batch_size=BATCH_SIZE,
            num_workers=4, pin_memory=True,
            collate_fn=collate_fn_padd
        )

    def validation_step(self, batch, batch_idx):
        loss, output, labels, label_lengths = self.step(batch)  # Compute loss for validation step
        self.val_losses.append(loss.item())  # Append validation loss for plotting
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # Log validation loss
        return {'val_loss': loss, 'output': output, 'labels': labels, 'label_lengths': label_lengths}

    def val_dataloader(self):
        d_params = Data.parameters  # Data parameters
        valid_dataset = Data(json_path='data/test.json', **d_params, valid=True)  # Validation dataset
        return DataLoader(
            dataset=valid_dataset, batch_size=BATCH_SIZE,
            num_workers=4, collate_fn=collate_fn_padd,
            pin_memory=True
        )

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # Compute average validation loss
        self.scheduler.step(avg_loss)
        self.log('avg_val_loss', avg_loss)

        # Evaluate WER and CER for a sample
        sample_output = outputs[0]['output']
        sample_labels = outputs[0]['labels']
        sample_label_lengths = outputs[0]['label_lengths']

        decoded_preds, decoded_targets = GreedyDecoder(sample_output, sample_labels, sample_label_lengths)
        wer_score = wer(decoded_targets[0], decoded_preds[0])
        cer_score = cer(decoded_targets[0], decoded_preds[0])

        print(f'Validation WER: {wer_score:.4f}, CER: {cer_score:.4f}')

        tensorboard_logs = {'val_loss': avg_loss, 'val_wer': wer_score, 'val_cer': cer_score}
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

def checkpoint_callback():
    return ModelCheckpoint(
        dirpath='checkpoints',  # Directory to save the checkpoints
        filename='{epoch:02d}-{val_loss:.2f}',  # Filename format for the checkpoints
        save_top_k=-1,  # Save all checkpoints
        verbose=True,  # Verbose output
        monitor='val_loss',  # Monitor validation loss
        mode='min'  # Save the model with the minimum validation loss
    )

def save_loss_graph(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss per Epoch')
    plt.savefig('graphs/loss_graph.png')  
    plt.close()

def main():
    os.makedirs('graphs', exist_ok=True)  
    os.makedirs('model', exist_ok=True)  
    os.makedirs('checkpoints', exist_ok=True)  
    os.makedirs('logs', exist_ok=True)  

    h_params = SpeechRecognition.hyper_parameters  
    model = SpeechRecognition(**h_params)  

    speech_module = SpeechModule(model)  

    logger = TensorBoardLogger('logs', name='speech_recognition')  
    trainer = Trainer(
        max_epochs=EPOCHS, gpus=1,  
        logger=logger, gradient_clip_val=1.0,
        val_check_interval=1.0,
        callbacks=[checkpoint_callback()],
    )
    trainer.fit(speech_module)  

    # Save the final model
    torch.save(model.state_dict(), 'model/model_mark_1.pth')

    # Save the loss graph
    save_loss_graph(speech_module.train_losses, speech_module.val_losses)

if __name__ == "__main__":
    main()  