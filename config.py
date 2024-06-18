# config.py

args = {
    'train_file': 'path/to/your/train_data.json',
    'valid_file': 'path/to/your/valid_data.json',
    'save_model_path': './checkpoints/',
    'logdir': './logs/',
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'data_workers': 4,
    'gpus': 1,  # Use 0 for CPU, 1 for GPU
    'nodes': 1,
    'valid_every': 0.5,
    'resume_from_checkpoint': None,  # Path to checkpoint if resuming
    'hparams_override': {},  # Any hyperparameters you want to override
    'dparams_override': {}  # Any data parameters you want to override
}