# training/sweeper.py

import wandb
from train import train_model, wandb_train

def run_sweep(entity_name, project_name, sweep_id=None):
    hyperparameters = {
        'batch_norm': {'values': [True, False]},
        'num_filters': {'values': [32, 64, 128, 256]},
        'filter_org': {'values': [0.5, 1, 2]},
        'dropout': {'values': [0.0, 0.5, 0.6, 0.4]},
        'data_augmentation': {'values': [True, False]},
        'num_epochs': {'values': [10, 20, 30]},
        'batch_size': {'values': [32, 64, 128]},
        'dense_layer': {'values': [32, 64, 128, 512]},
        'learning_rate': {'values': [0.001, 0.0001]},
        'kernel_size': {'values': [3, 5, 7]},
        'dense_activation': {'values': ['relu']},
        'conv_activation': {'values': ['relu']}
    }

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': hyperparameters,
    }
    
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, entity=entity_name, project=project_name)
        print("Initialized sweep with ID:", sweep_id)
        
    wandb.agent(sweep_id, function=wandb_train, entity=entity_name, project=project_name, count=50)
