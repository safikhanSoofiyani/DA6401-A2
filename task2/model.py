import torch
import torch.nn as nn
import torchvision.models as models

def get_activation(name):
    """Return activation given a string."""
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU()  # default

class CustomModel(nn.Module):
    def __init__(self, base_model, num_features, dense_layer, dropout, dense_activation, num_classes=10):
        """
        base_model   : pretrained model with its classifier removed
        num_features : output feature dimension from base_model
        dense_layer  : number of neurons in the custom dense layer
        dropout      : dropout probability
        dense_activation : activation function name to use in the dense layer
        """
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, dense_layer),
            get_activation(dense_activation),
            nn.Dropout(dropout),
            nn.Linear(dense_layer, num_classes)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

def build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers):
    """
    Builds a PyTorch model by:
      1. Loading a pretrained base network,
      2. Freezing its parameters,
      3. Optionally unfreezing the last N child modules (if trainable_layers > 0),
      4. Attaching a new classification head.

    model_name options: "ResNet50", "InceptionV3", "InceptionResNetV2", "Xception"
    """
    if model_name == 'ResNet50':
        base_model_full = models.resnet50(pretrained=True)
        num_features = base_model_full.fc.in_features
        # Remove the final FC layer (we take all layers except the last)
        base_model = nn.Sequential(*list(base_model_full.children())[:-1])
    elif model_name == 'InceptionV3':
        base_model_full = models.inception_v3(pretrained=True, aux_logits=False)
        num_features = base_model_full.fc.in_features
        base_model = nn.Sequential(*list(base_model_full.children())[:-1])
    elif model_name == 'InceptionResNetV2':
        import timm
        base_model_full = timm.create_model('inception_resnet_v2', pretrained=True)
        num_features = base_model_full.num_features
        base_model_full.reset_classifier(0)
        base_model = base_model_full
    else:  # default to Xception
        import timm
        base_model_full = timm.create_model('xception', pretrained=True)
        num_features = base_model_full.num_features
        base_model_full.reset_classifier(0)
        base_model = base_model_full

    # Freeze all base model parameters.
    for param in base_model.parameters():
        param.requires_grad = False

    # Optionally unfreeze the last few modules.
    if trainable_layers > 0:
        children = list(base_model.children())
        if len(children) > 0:
            num_children = len(children)
            layers_to_unfreeze = min(trainable_layers, num_children)
            for child in children[-layers_to_unfreeze:]:
                for param in child.parameters():
                    param.requires_grad = True
        else:
            # If the model doesn’t expose children in a sequential manner, unfreeze all.
            for param in base_model.parameters():
                param.requires_grad = True

    model = CustomModel(base_model, num_features, dense_layer, dropout, dense_activation)
    return model
