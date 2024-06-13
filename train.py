#train.py
import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from model import SpeechRecognition
from torch.utils.data import DataLoader
from dataset import Data, collate_fn_padd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class SpeechModule(LightningModule):
    def __init__(self, model, args):
        super(SpeechModule, self).__init__()
        self.model = model  # Speech recognition model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)  # CTC loss function
        self.args = args  # Arguments

    def forward(self, x, hidden):
        return self.model(x, hidden)  # Forward pass

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)  # Optimizer
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
        output = F.log_softmax(output, dim=2)  # Log softmax activation
        loss = self.criterion(output, labels, input_lengths, label_lengths)  # Compute loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)  # Compute loss for training step
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # Log training loss
        return {'loss': loss}

    def train_dataloader(self):
        d_params = Data.parameters  # Data parameters
        d_params.update(self.args.dparams_override)  # Override parameters
        train_dataset = Data(json_path=self.args.train_file, **d_params)  # Training dataset
        return DataLoader(
            dataset=train_dataset, batch_size=self.args.batch_size,
            num_workers=self.args.data_workers, pin_memory=True,
            collate_fn=collate_fn_padd
        )

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)  # Compute loss for validation step
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # Log validation loss
        return {'val_loss': loss}

    def val_dataloader(self):
        d_params = Data.parameters  # Data parameters
        d_params.update(self.args.dparams_override)  # Override parameters
        valid_dataset = Data(json_path=self.args.valid_file, **d_params, valid=True)  # Validation dataset
        return DataLoader(
            dataset=valid_dataset, batch_size=self.args.batch_size,
            num_workers=self.args.data_workers, collate_fn=collate_fn_padd,
            pin_memory=True
        )

def checkpoint_callback(args):
    return ModelCheckpoint(
        dirpath=os.path.dirname(args.save_model_path),  # Directory to save the model
        filename=os.path.basename(args.save_model_path),  # Filename for the saved model
        save_top_k=1,  # Save only the best model
        verbose=True,  # Verbose output
        monitor='val_loss',  # Monitor validation loss
        mode='min'  # Save the model with the minimum validation loss
    )

def main(args):
    h_params = SpeechRecognition.hyper_parameters  # Default hyperparameters
    h_params.update(args.hparams_override)  # Override hyperparameters
    model = SpeechRecognition(**h_params)  # Initialize model

    if args.load_model_from:
        speech_module = SpeechModule.load_from_checkpoint(args.load_model_from, model=model, args=args)  # Load pre-trained model
    else:
        speech_module = SpeechModule(model, args)  # Initialize new model

    logger = TensorBoardLogger(args.logdir, name='speech_recognition')  # TensorBoard logger
    trainer = Trainer(
        max_epochs=args.epochs, gpus=args.gpus,
        num_nodes=args.nodes, logger=logger, gradient_clip_val=1.0,
        val_check_interval=args.valid_every,
        checkpoint_callback=checkpoint_callback(args),
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.fit(speech_module)  # Train the model

if __name__ == "__main__":
    parser = ArgumentParser()

    # Distributed training setup
    parser.add_argument('-n', '--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('-w', '--data_workers', default=0, type=int, help='Number of data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='Distributed backend to use')

    # Training and validation files
    parser.add_argument('--train_file', required=True, type=str, help='JSON file for training data')
    parser.add_argument('--valid_file', required=True, type=str, help='JSON file for validation data')
    parser.add_argument('--valid_every', default=1000, type=int, help='Validation frequency')

    # Paths for saving models and logs
    parser.add_argument('--save_model_path', required=True, type=str, help='Path to save the model')
    parser.add_argument('--load_model_from', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to resume training from checkpoint')
    parser.add_argument('--logdir', default='tb_logs', type=str, help='Path to save logs')

    # Training parameters
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--pct_start', default=0.3, type=float, help='Percentage of growth phase in one cycle')
    parser.add_argument('--div_factor', default=100, type=int, help='Div factor for one cycle')
    parser.add_argument("--hparams_override", default="{}", type=str, help='Override hyperparameters')
    parser.add_argument("--dparams_override", default="{}", type=str, help='Override data parameters')

    args = parser.parse_args()
    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_override)

    if args.save_model_path:
        save_dir = os.path.dirname(args.save_model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create directory if it does not exist

    main(args)  # Run the main function