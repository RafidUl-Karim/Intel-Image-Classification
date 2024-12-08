import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_datasets(config):
    # Define transforms for the training and testing datasets
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5 if config["training"]["augmentation"]["horizontal_flip"] else 0.0),
        transforms.RandomRotation(degrees=config["training"]["augmentation"]["rotation_range"]),
        transforms.RandomResizedCrop(size=config["model"]["input_shape"][:2], scale=(1.0 - config["training"]["augmentation"]["zoom_range"], 1.0)),
        transforms.ColorJitter(brightness=config["training"]["augmentation"]["brightness_range"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common normalization for pretrained models
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(config["model"]["input_shape"][:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the datasets using ImageFolder
    train_dataset = datasets.ImageFolder(
        root=config["dataset"]["train_data_path"],
        transform=train_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=config["dataset"]["test_data_path"],
        transform=test_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True  # Helpful for GPU training
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader
