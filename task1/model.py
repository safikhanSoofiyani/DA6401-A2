# models/cnn_model.py

import torch
import torch.nn as nn
import random
import numpy as np

from data_loader import IMG_HEIGHT, IMG_WIDTH
# utils/utils.py



def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def get_activation_layer(act_name):
    act_name = act_name.lower()
    if act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError(f"Activation function {act_name} is not implemented.")


# Define your CNN model; note: PyTorch uses channels-first format.
class CNNModel(nn.Module):
    def __init__(self, conv_activations, dense_activation, num_filters, conv_filter_sizes, pool_filter_sizes, 
                 batch_norm, dense_layer, dropout, img_shape=(3, IMG_HEIGHT, IMG_WIDTH)):
        super(CNNModel, self).__init__()
        self.batch_norm = batch_norm
        layers = []
        in_channels = img_shape[0]
        for i in range(5):
            out_channels = int(num_filters[i])
            kernel_size = conv_filter_sizes[i]
            pool_size = pool_filter_sizes[i]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(get_activation_layer(conv_activations[i]))
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=2))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        # Dynamically compute the flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, *img_shape)
            conv_out = self.conv(dummy)
            flattened_dim = conv_out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_dim, dense_layer)
        self.fc_act = get_activation_layer(dense_activation)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_layer, 10)  # 10 classes output

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc_act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

