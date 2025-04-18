import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from data_loader import get_data, get_augmented_data, get_test_data, get_test_augmented_data
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_loop(model, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def wandb_train():
    # Get hyperparameters from wandb config.
    wandb.init()
    config = wandb.config
    model_name = config.model_name
    data_augmentation = config.data_augmentation
    dense_layer = config.dense_layer
    dropout = config.dropout
    trainable_layers = config.trainable_layers
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    dense_activation = "relu"
    
    run_name = f"Model_{model_name}_trainable_{trainable_layers}_Data_aug_{data_augmentation}_dropout_{dropout}_dense_layer_{dense_layer}"
    wandb.run.name = run_name
    
    # In our folder structure, the training images are in "/projects/data/llmteam/safi/dl_course/assignment_2/data/inaturalist_12K/train"
    train_dir = "/projects/data/llmteam/safi/dl_course/assignment_2/data/inaturalist_12K/train"
    # We use the same folder (and then split) for validation.
    if data_augmentation:
        train_loader, val_loader = get_augmented_data(train_dir, batch_size=batch_size)
    else:
        train_loader, val_loader = get_data(train_dir, batch_size=batch_size)
    
    model = build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_loop(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate_loop(model, val_loader, criterion, epoch)
        wandb.log({
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save best model
            torch.save(model.state_dict(), "best_model.pth")
    return model

def sweeper(entity_name, project_name):
    hyperparameters = {
        "model_name": {"values": ["ResNet50"]},
        "data_augmentation": {"values": [True, False]},
        "dense_layer": {"values": [64, 128, 256, 512]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
        "trainable_layers": {"values": [0, 10, 15, 20]},
        "batch_size": {"values": [64, 128]},
        "num_epochs": {"values": [5, 10, 15]}
    }
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "parameters": hyperparameters
    }
    sweep_id = wandb.sweep(sweep_config, entity=entity_name, project=project_name)
    wandb.agent(sweep_id, function=wandb_train)

def testing(entity_name, project_name):
    """
    This function sets the best hyperparameters, trains on the full train data,
    and then evaluates on the test set.
    """
    best_hyperparameters = {
        "model_name": "Xception",
        "data_augmentation": True,
        "dense_layer": 128,
        "dropout": 0.3,
        "trainable_layers": 20,
        "batch_size": 64,
        "num_epochs": 15
    }
    wandb.init(config=best_hyperparameters, project=project_name, entity=entity_name, name="test run - partb")
    config = wandb.config
    model_name = config.model_name
    data_augmentation = config.data_augmentation
    dense_layer = config.dense_layer
    dropout = config.dropout
    trainable_layers = config.trainable_layers
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    dense_activation = "relu"
    
    train_dir = "/projects/data/llmteam/safi/dl_course/assignment_2/data/inaturalist_12K/train"
    test_dir = "/projects/data/llmteam/safi/dl_course/assignment_2/data/inaturalist_12K/val"
    
    if data_augmentation:
        train_loader, test_loader = get_test_augmented_data(train_dir, test_dir, batch_size=batch_size)
    else:
        train_loader, test_loader = get_test_data(train_dir, test_dir, batch_size=batch_size)
    
    model = build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    # Train on the full training set (no validation split)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_loop(model, train_loader, criterion, optimizer, epoch)
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "epoch": epoch})
    
    # Save the trained model.
    torch.save(model.state_dict(), "best_pretrained_model.pth")
    test_loss, test_acc = validate_loop(model, test_loader, criterion, "Test")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    wandb.finish()
    return model

def testing_cnn(model_name, dropout, data_augmentation, num_epochs, batch_size, dense_layer, learning_rate, trainable_layers, dense_activation):
    """
    This function trains the model using user‑provided hyperparameters and then evaluates it on test data.
    It mirrors the original "testing_cnn" functionality.
    """
    train_dir = "/projects/data/llmteam/safi/dl_course/assignment_2/data/inaturalist_12K/train"
    test_dir = "/projects/data/llmteam/safi/dl_course/assignment_2/data/inaturalist_12K/val"
    
    if data_augmentation == "True":
        train_loader, test_loader = get_test_augmented_data(train_dir, test_dir, batch_size=batch_size)
    else:
        train_loader, test_loader = get_test_data(train_dir, test_dir, batch_size=batch_size)
    
    model = build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)
    model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_loop(model, train_loader, criterion, optimizer, epoch)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
    
    test_loss, test_acc = validate_loop(model, test_loader, criterion, "Test")
    print("The test Loss is :", test_loss)
    print("The test Accuracy is:", test_acc)
    
    torch.save(model.state_dict(), "best_model_pretrained.pth")
    return model
