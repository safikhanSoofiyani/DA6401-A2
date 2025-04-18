
import os
import sys
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

# Uncomment if using wandb
# import wandb

IMG_HEIGHT, IMG_WIDTH, CHANNELS = 256, 256, 3

# Global flag for debug prints
DEBUG_MODE = False

class TransformedSubset(Subset):
    def __getitem__(self, idx):
        # Get the original sample from the parent dataset
        image, label = self.dataset[self.indices[idx]]
        if DEBUG_MODE:
            print(f"[TransformedSubset] Index {idx} - Before transformation: type={type(image)}")
        # Try to apply the transform.
        try:
            transformed = self.transform(image)
        except Exception as e:
            print(f"[TransformedSubset] Error applying transform at index {idx}: {e}")
            # Fallback: force conversion with ToTensor.
            transformed = transforms.ToTensor()(image)
        # If the result is not a tensor, force conversion.
        if not isinstance(transformed, torch.Tensor):
            print(f"[TransformedSubset] Warning: Index {idx} transform did not return tensor; forcing conversion.")
            try:
                transformed = transforms.ToTensor()(transformed)
            except Exception as e:
                print(f"[TransformedSubset] Error forcing tensor conversion at index {idx}: {e}")
                raise e
        if DEBUG_MODE:
            print(f"[TransformedSubset] Index {idx} - After transformation: type={type(transformed)}")
        return transformed, label




def get_data(batch_size, num_workers=20):
    """
    Load train data from a directory and split into training and validation sets (90/10).
    No augmentation is applied (only resizing & conversion to tensor).
    """
    train_dir = r"data/inaturalist_12K/train"

    transform_basic = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),  # Converts and scales pixel values to [0,1]
    ])
    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform_basic)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_augmented_data(batch_size, seed=42, num_workers=20):
    '''
    Returns PyTorch DataLoader objects for training and validation datasets
    using 10% of the data for validation with augmentations applied to training data.

    Parameters:
    - batch_size (int): Batch size for the dataloaders.
    - seed (int): Random seed for reproducibility.

    Returns:
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    '''

    path = r"data/inaturalist_12K/train"

    # Transforms for training and validation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load the full dataset with training transform (we will later split it)
    full_dataset = ImageFolder(root=path, transform=None)

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Calculate split sizes
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Apply transforms post-split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_test_data(batch_size, num_workers=2):
    """
    Load training and test data (test data comes from the given 'val' directory). No augmentation is applied.
    """
    train_dir = r"data/inaturalist_12K/train"
    test_dir  = r"data/inaturalist_12K/val"

    transform_basic = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_basic)
    test_dataset  = datasets.ImageFolder(root=test_dir, transform=transform_basic)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_augmented_test_data(batch_size, num_workers=2):
    """
    Load training data with augmentation and test data without augmentation.
    """
    train_dir = r"data/inaturalist_12K/train"
    test_dir  = r"data/inaturalist_12K/val"

    transform_train = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset  = datasets.ImageFolder(root=test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# ----------------------------- Additional Debug Functions ----------------------------- #

def debug_transform():
    """
    Load one image from the base dataset (without any transform), apply the training transformation, and print type info.
    """
    train_dir = r"data/inaturalist_12K/train"
    base_dataset = datasets.ImageFolder(root=train_dir, transform=None)
    # Use the same transform as in get_augmented_data
    transform_train = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # Get the first sample
    image, label = base_dataset[0]
    print("[debug_transform] Before transformation:", type(image))
    try:
        image_transformed = transform_train(image)
        print("[debug_transform] After transformation:", type(image_transformed))
    except Exception as e:
        print("[debug_transform] Error applying transform:", e)


def test_loader(loader_func, batch_size=4, loader_name="Loader", num_workers=0):
    """
    Helper function to test a given loader by iterating over one batch and printing info.
    Using num_workers=0 is recommended for debugging purposes.
    """
    print(f"--- Testing {loader_name} ---")
    try:
        loaders = loader_func(batch_size, num_workers=num_workers)
    except Exception as e:
        print(f"[{loader_name}] Error in loader function: {e}")
        return

    if isinstance(loaders, (list, tuple)):
        for i, loader in enumerate(loaders):
            print(f">> Testing loader #{i+1}")
            try:
                for batch in loader:
                    images, labels = batch
                    print("Image type: ", type(images))
                    print("Image shape: ", images.shape)
                    print("Labels type: ", type(labels))
                    print("Labels shape: ", labels.shape)
                    print("Image min, max: ", images.min().item(), images.max().item())
                    break  # Only test one batch per loader
            except Exception as e:
                print(f"[{loader_name} - loader #{i+1}] Error during iteration: {e}")
    else:
        try:
            for batch in loaders:
                images, labels = batch
                print("Image type: ", type(images))
                print("Image shape: ", images.shape)
                print("Labels type: ", type(labels))
                print("Labels shape: ", labels.shape)
                print("Image min, max: ", images.min().item(), images.max().item())
                break
        except Exception as e:
            print(f"[{loader_name}] Error during iteration: {e}")


def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser(description="Test DataLoader functions.")
    parser.add_argument('--debug', action="store_true", help="Enable debug mode for extra information.")
    args = parser.parse_args()
    if args.debug:
        DEBUG_MODE = True
        num_workers = 0
    else:
        num_workers = 2

    print("\n>>> Running debug_transform() to check individual transformation:")
    debug_transform()

    print("\n>>> Testing get_data:")
    test_loader(get_data, batch_size=4, loader_name="get_data", num_workers=num_workers)

    print("\n>>> Testing get_augmented_data:")
    test_loader(get_augmented_data, batch_size=4, loader_name="get_augmented_data", num_workers=num_workers)

    print("\n>>> Testing get_test_data:")
    test_loader(get_test_data, batch_size=4, loader_name="get_test_data", num_workers=num_workers)

    print("\n>>> Testing get_augmented_test_data:")
    test_loader(get_augmented_test_data, batch_size=4, loader_name="get_augmented_test_data", num_workers=num_workers)


if __name__ == '__main__':
    main()

