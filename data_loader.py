import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_data(data_dir, batch_size=64, validation_split=0.1, image_size=(256, 256)):
    """
    Creates training and validation loaders without additional augmentation.
    Uses a random split from the directory provided.
    """
    basic_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=basic_transform)
    n_total = len(full_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def get_augmented_data(data_dir, batch_size=64, validation_split=0.1, image_size=(256, 256)):
    """
    Creates training and validation loaders with data augmentation for training.
    (The validation set is kept as the basic rescaled version.)
    """
    # Augmentation for training: random translation, horizontal flip, and zoom (scale)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.2, 0.2), 
            scale=(0.8, 1.2),
            fill=0  # "nearest" fill mode approximation
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    # To make sure that train and validation get the same split indices,
    # load the full dataset and then override the transform attribute.
    full_dataset = datasets.ImageFolder(root=data_dir)
    n_total = len(full_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val])
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def get_test_data(train_dir, test_dir, batch_size=64, image_size=(256, 256)):
    """
    Creates loaders for training and test data (without augmentation).
    Note: here the training and test folders are expected to be separate.
    """
    basic_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root=train_dir, transform=basic_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=basic_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def get_test_augmented_data(train_dir, test_dir, batch_size=64, image_size=(256,256)):
    """
    Creates loaders for training (augmented) and test data.
    Augmentation is applied only to the training loader.
    """
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.2, 0.2), 
            scale=(0.8, 1.2),
            fill=0
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader
