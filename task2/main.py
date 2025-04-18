import argparse
from train import testing_cnn, sweeper, testing
import wandb
import torch
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default="safikhan", required=False, help="WandB entity name")
    parser.add_argument('--project', type=str, default="DA6401_Assignment_2", required=False, help="WandB project name")
    parser.add_argument('--sweep', type=str, default="no", required=False, help="Enter 'yes' for sweep mode or 'no' for normal training")
    parser.add_argument('--model', type=str, default="ResNet50", required=False, help="Pretrained model to use: InceptionV3, ResNet50, InceptionResNetV2, or Xception")
    parser.add_argument('--dropout', type=float, default=0.3, required=False, help="Dropout (float value)")
    parser.add_argument('--dataAugment', type=bool, default=False, required=False, help="Data Augmentation: True or False")
    parser.add_argument('--numEpochs', type=int, default=15, required=False, help="Number of Epochs (integer)")
    parser.add_argument('--batchSize', type=int, default=64, required=False, help="Batch size (integer)")
    parser.add_argument('--denseLayer', type=int, default=128, required=False, help="Dense layer size (integer)")
    parser.add_argument('--learningRate', type=float, default=0.0001, required=False, help="Learning rate (float)")
    parser.add_argument('--trainLayers', type=int, default=20, required=False, help="Number of trainable layers (integer)")
    parser.add_argument('--denseAct', type=str, default="relu", required=False, help="Activation function for the dense layer (string)")
    args = parser.parse_args()

    if args.sweep.lower() == "no":
        if args.model not in ["InceptionV3", "ResNet50", "InceptionResNetV2", "Xception"]:
            print("Please enter a valid model name: InceptionV3, ResNet50, InceptionResNetV2, or Xception")
            exit()

        model = testing_cnn(
            args.model, args.dropout, args.dataAugment, args.numEpochs, args.batchSize,
            args.denseLayer, args.learningRate, args.trainLayers, args.denseAct
        )
    else:
        entity_name = args.entity
        project_name = args.project
        sweeper(entity_name, project_name)
        testing(entity_name, project_name)

if __name__ == "__main__":
    # Set random seeds based on hash values
    random.seed(hash("seriously you compete with me") % (2**32 - 1))
    np.random.seed(hash("i am mohammed safi") % (2**32 - 1))
    import torch
    torch.manual_seed(hash("ur rahman khan") % (2**32 - 1))
    main()
