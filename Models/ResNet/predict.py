import os
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config import load_config
from model import create_model
from torch.utils.data import DataLoader

def predict_images():
    # Load the configuration
    config = load_config()

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the model and load the checkpoint
    model, _, _ = create_model(config)
    model.load_state_dict(torch.load(config["logging"]["checkpoint_path"], map_location=device))
    model.to(device)
    model.eval()

    # Define transformations for the test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training transforms
    ])

    # Load the test dataset using ImageFolder (assuming the directory structure is correct)
    test_path = config["dataset"]["test_data_path"]
    class_labels = config["dataset"]["class_labels"]

    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    # For tracking metrics
    all_true_labels = []
    all_pred_labels = []

    print("Running Predictions...")
    progress_bar = tqdm(total=len(test_loader), desc="Predicting", unit="batch")

    # Predict for each image in the test dataset
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1)

        # Store true labels and predicted labels
        all_true_labels.extend(labels.cpu().numpy())
        all_pred_labels.extend(preds.cpu().numpy())

        progress_bar.update(1)

    progress_bar.close()

    # Calculate metrics
    test_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    print(f"Test Accuracy: {test_accuracy:.3f}")

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels, target_names=class_labels))

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Accuracy Score (Manual Check)
    print("-" * 60)  # Replacing Unicode character with a regular dash
    print(f"Accuracy Score (Manual Check): {test_accuracy:.3f}")
    print("-" * 60)  # Replacing Unicode character with a regular dash

if __name__ == "__main__":
    predict_images()
