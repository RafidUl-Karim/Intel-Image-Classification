import os
import torch
from tqdm import tqdm
from data_loader import prepare_datasets
from model import create_model
from config import load_config

def train_model():
    # Load the configuration
    config = load_config()
    
    # Prepare the datasets
    train_loader, test_loader = prepare_datasets(config)
    
    # Create the model, optimizer, and loss function
    model, optimizer, loss_function = create_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Ensure checkpoint directory exists
    checkpoint_path = config["logging"]["checkpoint_path"]
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early stopping parameters
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    # Training loop
    epochs = config["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(total=len(train_loader), desc="Training", unit="batch")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "accuracy": f"{(train_correct / train_total):.4f}"
            })
            train_bar.update(1)
        train_bar.close()
        
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(total=len(test_loader), desc="Validation", unit="batch")
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "val_accuracy": f"{(val_correct / val_total):.4f}"
                })
                val_bar.update(1)
        val_bar.close()
        
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_accuracy = val_correct / val_total
        
        # Summary for the epoch
        print(f"Epoch {epoch}: "
              f"Loss={epoch_train_loss:.4f}, Accuracy={epoch_train_accuracy:.4f}, "
              f"Val_Loss={epoch_val_loss:.4f}, Val_Accuracy={epoch_val_accuracy:.4f}")
        
        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model improved and saved to {checkpoint_path}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
        
        if epochs_no_improve >= patience:
            print(f"Stopping early after {patience} epochs without improvement.")
            break
    
    print("Training completed.")

if __name__ == "__main__":
    train_model()
