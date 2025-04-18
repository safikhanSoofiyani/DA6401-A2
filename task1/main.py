# main.py

import argparse
import torch

from train import train_model  # or testing functions as needed
from sweep import run_sweep

DEFAULT_CONFIG = {
    'batch_norm': True,
    'num_filters': 32,
    'filter_org': 2,
    'dropout': 0.4,
    'data_augmentation': True,
    'num_epochs': 10,
    'batch_size': 128,
    'dense_layer': 512,
    'learning_rate': 0.0001,
    'kernel_size': 3,
    'dense_activation': 'relu',
    'conv_activation': 'relu'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run CNN training/testing.")
    parser.add_argument('--sweep', type=str, required=True, help="Enter 'yes' for sweep mode or 'no' for normal training.")
    parser.add_argument('--sweep_id', type=str, default=None, help="Sweep ID for wandb sweep.")
    parser.add_argument('--entity_name', type=str, default="safikhan", help="Entity name for wandb.")
    parser.add_argument('--project_name', type=str, default="DA6401_Assignment_2", help="Project name for wandb.")
    # Add all additional arguments here if not in sweep mode
    parser.add_argument('--batchNorm', type=bool, help="Batch Normalization flag.")
    parser.add_argument('--numFilters', type=int, help="Number of filters for the conv layers.")
    parser.add_argument('--filterOrg', type=float, help="Filter organization multiplier.")
    parser.add_argument('--dropout', type=float, help="Dropout rate.")
    parser.add_argument('--dataAugment', type=bool, help="Data augmentation flag.")
    parser.add_argument('--numEpochs', type=int, help="Number of training epochs.")
    parser.add_argument('--batchSize', type=int, help="Batch size.")
    parser.add_argument('--denseLayer', type=int, help="Size of the dense layer.")
    parser.add_argument('--learningRate', type=float, help="Learning rate for optimization.")
    parser.add_argument('--kernelSize', type=int, help="Kernel size for conv layers.")
    parser.add_argument('--denseAct', type=str, help="Dense layer activation function.")
    parser.add_argument('--convAct', type=str, help="Convolution layer activation function.")
    return parser.parse_args()

def main():
    args = parse_args()
    # Use GPU if available
    DEFAULT_CONFIG['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.sweep.lower() == 'no':
        # Merge command-line args into DEFAULT_CONFIG for a single run
        config = DEFAULT_CONFIG.copy()
        # Override with any specified command-line arguments
        if args.batchNorm is not None: config['batch_norm'] = args.batchNorm
        if args.numFilters is not None: config['num_filters'] = args.numFilters
        if args.filterOrg is not None: config['filter_org'] = args.filterOrg
        if args.dropout is not None: config['dropout'] = args.dropout
        if args.dataAugment is not None: config['data_augmentation'] = args.dataAugment
        if args.numEpochs is not None: config['num_epochs'] = args.numEpochs
        if args.batchSize is not None: config['batch_size'] = args.batchSize
        if args.denseLayer is not None: config['dense_layer'] = args.denseLayer
        if args.learningRate is not None: config['learning_rate'] = args.learningRate
        if args.kernelSize is not None: config['kernel_size'] = args.kernelSize
        if args.denseAct is not None: config['dense_activation'] = args.denseAct
        if args.convAct is not None: config['conv_activation'] = args.convAct

        train_model(config)
    else:
        run_sweep(entity_name=args.entity_name, project_name=args.project_name, sweep_id=args.sweep_id)

if __name__ == '__main__':
    main()
