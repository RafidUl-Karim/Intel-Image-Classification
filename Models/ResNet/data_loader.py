def prepare_datasets(config):
    from torchvision import datasets, transforms
    import torch

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if config["training"]["augmentation"]["horizontal_flip"] else None,
            transforms.RandomRotation(config["training"]["augmentation"]["rotation_range"]),
            transforms.RandomResizedCrop(224, scale=(1 - config["training"]["augmentation"]["zoom_range"], 1)),
            transforms.ColorJitter(brightness=config["training"]["augmentation"]["brightness_range"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=config["dataset"]["train_data_path"],
        transform=data_transforms['train']
    )
    test_dataset = datasets.ImageFolder(
        root=config["dataset"]["test_data_path"],
        transform=data_transforms['test']
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )

    return train_loader, test_loader
