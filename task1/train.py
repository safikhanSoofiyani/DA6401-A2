# training/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import CNNModel
from data_loader import get_data, get_augmented_data, get_test_data, get_augmented_test_data
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(config):
    wandb.init(config=config)
    config = wandb.config

    # Set up activations, filter parameters, etc.
    conv_activations = [config.conv_activation] * 5
    dense_activation = config.dense_activation

    num_filters = []
    filters = config.num_filters
    for _ in range(5):
        num_filters.append(filters)
        filters *= config.filter_org

    conv_filter_sizes = [(config.kernel_size, config.kernel_size)] * 5
    pool_filter_sizes = [(2, 2)] * 5

    from model import set_seed
    set_seed(42)

    model = CNNModel(conv_activations, dense_activation, num_filters, conv_filter_sizes,
                     pool_filter_sizes, config.batch_norm, config.dense_layer, config.dropout)
    model.to(device)

    if config.data_augmentation:
        train_loader, val_loader = get_augmented_data(config.batch_size)
    else:
        train_loader, val_loader = get_data(config.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(config.num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        print(f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    wandb.finish()
    return model

def wandb_train():
    wandb.init()
    config = wandb.config

    # Set up activations, filter parameters, etc.
    conv_activations = [config.conv_activation] * 5
    dense_activation = config.dense_activation

    num_filters = []
    filters = config.num_filters
    for _ in range(5):
        num_filters.append(filters)
        filters *= config.filter_org

    conv_filter_sizes = [(config.kernel_size, config.kernel_size)] * 5
    pool_filter_sizes = [(2, 2)] * 5

    from model import set_seed
    set_seed(42)

    model = CNNModel(conv_activations, dense_activation, num_filters, conv_filter_sizes,
                     pool_filter_sizes, config.batch_norm, config.dense_layer, config.dropout)
    model.to(device)

    if config.data_augmentation:
        train_loader, val_loader = get_augmented_data(config.batch_size)
    else:
        train_loader, val_loader = get_data(config.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    run_name = f"{config.batch_norm}-{config.num_filters}-{config.filter_org}-{config.dropout}-{config.data_augmentation}-{config.num_epochs}-{config.batch_size}-{config.dense_layer}-{config.learning_rate}-{config.kernel_size}-{config.dense_activation}-{config.conv_activation}"
    wandb.run.name = run_name

    for epoch in tqdm(range(config.num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        print(f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    wandb.run.save()
    wandb.finish()
    return model

def testing(entity_name, project_name):
    """
    Train the best model on the complete training set and evaluate it on a separate test set.
    Uses best hyperparameters (hard-coded) and logs test metrics to wandb.
    """
    best_hyperparameters = {
        'batch_norm': True,
        'num_filters': 32,
        'filter_org': 2,
        'dropout': 0.4,
        'data_augmentation': False,
        'num_epochs': 10,
        'batch_size': 128,
        'dense_layer': 512,
        'learning_rate': 0.0001,
        'kernel_size': 3,
        'dense_activation': 'relu',
        'conv_activation': 'relu'
    }
    wandb.init(config=best_hyperparameters, project=project_name, entity=entity_name)
    config = wandb.config

    conv_activations = [config.conv_activation] * 5
    dense_activation = config.dense_activation

    num_filters = []
    filters = config.num_filters
    for i in range(5):
        num_filters.append(filters)
        filters = filters * config.filter_org

    conv_filter_sizes = [(config.kernel_size, config.kernel_size)] * 5
    pool_filter_sizes = [(2, 2)] * 5

    model = CNNModel(conv_activations, dense_activation, num_filters, conv_filter_sizes,
                     pool_filter_sizes, config.batch_norm, config.dense_layer, config.dropout)
    model.to(device)
    print(model)

    # Get data: here we use non-augmented data (as per best_hyperparameters)
    train_loader, test_loader = get_test_data(config.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_epochs = config.num_epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_loss /= test_total
    test_acc = test_correct / test_total
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
    wandb.finish()

    # Save the model
    torch.save(model.state_dict(), "best_model.pth")
    return model


def testing_cnn(batch_norm, num_filters, filter_org, dropout, data_augmentation, num_epochs,
                batch_size, dense_layer, learning_rate, kernel_size, dense_activation, conv_activation):
    """
    Train and evaluate the model based on user-specified hyperparameters (without wandb logging).
    """
    conv_activations = [conv_activation] * 5

    # Compute list of filters for each conv layer
    num_filter_list = []
    filters = num_filters
    for i in range(5):
        num_filter_list.append(filters)
        filters = filters * filter_org

    conv_filter_sizes = [(kernel_size, kernel_size)] * 5
    pool_filter_sizes = [(2, 2)] * 5

    model = CNNModel(conv_activations, dense_activation, num_filter_list, conv_filter_sizes,
                     pool_filter_sizes, batch_norm, dense_layer, dropout)
    model.to(device)
    print(model)

    # Get the data
    if data_augmentation:
        train_loader, test_loader = get_augmented_test_data(batch_size)
    else:
        train_loader, test_loader = get_test_data(batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_loss /= test_total
    test_acc = test_correct / test_total
    print("The test Loss is :", test_loss)
    print("The test Accuracy is:", test_acc)

    # Save the model
    torch.save(model.state_dict(), "best_model.pth")
    return model

#6na93evo
