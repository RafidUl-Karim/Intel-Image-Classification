from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

def create_model(config):
    # Load ResNet50 with or without pretrained weights
    weights = ResNet50_Weights.DEFAULT if config["model"]["pretrained"] else None
    model = resnet50(weights=weights)

    # Modify the final fully connected layer to match the number of output classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config["model"]["output_classes"])

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_function = nn.CrossEntropyLoss()

    return model, optimizer, loss_function
