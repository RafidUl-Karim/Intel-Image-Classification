import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from config import load_config
from model import create_model

def predict_images():
    # Load the configuration
    config = load_config()
    
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create the model architecture and load the checkpoint
    model, _, _ = create_model(config)
    model.load_state_dict(torch.load(config["logging"]["checkpoint_path"], map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare image transformations
    input_size = config["model"]["input_shape"][:2]
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get the prediction dataset path and class labels
    pred_path = config["dataset"]["pred_data_path"]
    class_labels = config["dataset"]["class_labels"]
    image_files = os.listdir(pred_path)
    
    print("Running Predictions...")
    progress_bar = tqdm(total=len(image_files), desc="Predicting", unit="image")
    
    # Predict each image
    for img_file in image_files:
        img_path = os.path.join(pred_path, img_file)
        
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
        
        # Perform prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Get the predicted class
        class_idx = np.argmax(probabilities)
        class_label = class_labels[class_idx]
        
        print(f"Image: {img_file}, Predicted Class: {class_label}")
        progress_bar.update(1)
    
    progress_bar.close()

if __name__ == "__main__":
    predict_images()
