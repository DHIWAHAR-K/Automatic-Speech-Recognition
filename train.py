#train.py
import os
import torch
import torch.nn as nn
from config import args
import torch.optim as optim
from model import SpeechRecognition
from torch.nn import functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from dataset import Data, collate_fn_padd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

class SpeechModule(LightningModule):
    def __init__(self, model, args):
        super(SpeechModule, self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.50, patience=6)
        return [self.optimizer], [self.scheduler]

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args['dparams_override'])
        train_dataset = Data(json_path=self.args['train_file'], **d_params)
        return DataLoader(dataset=train_dataset, batch_size=self.args['batch_size'],
                          num_workers=self.args['data_workers'], pin_memory=True,
                          collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args['dparams_override'])
        test_dataset = Data(json_path=self.args['valid_file'], **d_params, valid=True)
        return DataLoader(dataset=test_dataset, batch_size=self.args['batch_size'],
                          num_workers=self.args['data_workers'], collate_fn=collate_fn_padd,
                          pin_memory=True)

def checkpoint_callback(args):
    return ModelCheckpoint(
        filepath=args['save_model_path'],
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

def main(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args['hparams_override'])
    model = SpeechRecognition(**h_params)

    if args['load_model_from']:
        speech_module = SpeechModule.load_from_checkpoint(args['load_model_from'], model=model, args=args)
    else:
        speech_module = SpeechModule(model, args)

    logger = TensorBoardLogger(args['logdir'], name='speech_recognition')

    trainer = Trainer(
        max_epochs=args['epochs'], gpus=args['gpus'], num_nodes=args['nodes'], distributed_backend=None,
        logger=logger, gradient_clip_val=1.0, val_check_interval=args['valid_every'],
        checkpoint_callback=checkpoint_callback(args), resume_from_checkpoint=args['resume_from_checkpoint']
    )
    trainer.fit(speech_module)

if __name__ == "__main__":
    main(args)