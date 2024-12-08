import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

def create_model(config):
    # Load the ResNet50 model
    base_model = models.resnet50(pretrained=config["model"]["pretrained"])
    
    # Modify the base model to exclude the fully connected (FC) layer
    num_features = base_model.fc.in_features  # Get the number of features in the last layer
    base_model.fc = nn.Identity()  # Replace the FC layer with an identity layer
    
    # Create a custom classifier
    classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, config["model"]["output_classes"]),
        nn.Softmax(dim=1) if config["model"]["activation"] == "softmax" else nn.Identity()
    )
    
    # Combine the base model and custom classifier
    model = nn.Sequential(
        base_model,
        nn.AdaptiveAvgPool2d((1, 1)),  # Equivalent to GlobalAveragePooling2D in TensorFlow
        nn.Flatten(),
        classifier
    )
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_function = nn.CrossEntropyLoss()  # CrossEntropyLoss is suitable for multi-class classification
    
    return model, optimizer, loss_function
