import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from data_loader import prepare_datasets
from model import create_model
from config import load_config

def train_model():
    config = load_config()
    
    train_gen, test_gen = prepare_datasets(config)
    model = create_model(config)
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(config["logging"]["checkpoint_path"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # EarlyStopping parameters
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Custom training loop with progress bar
    epochs = config["training"]["epochs"]
    steps_per_epoch = len(train_gen)
    val_steps = len(test_gen)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Training progress bar
        train_bar = tqdm(total=steps_per_epoch, desc="Training", unit="batch")
        train_loss, train_accuracy = [], []
        
        for x, y in train_gen:
            metrics = model.train_on_batch(x, y)
            train_loss.append(metrics[0])  # loss
            train_accuracy.append(metrics[1])  # accuracy
            
            train_bar.set_postfix({
                "loss": f"{metrics[0]:.4f}",
                "accuracy": f"{metrics[1]:.4f}"
            })
            train_bar.update(1)
        
        train_bar.close()
        
        # Validation progress bar
        val_bar = tqdm(total=val_steps, desc="Validation", unit="batch")
        val_loss, val_accuracy = [], []
        
        for x, y in test_gen:
            metrics = model.test_on_batch(x, y)
            val_loss.append(metrics[0])  # loss
            val_accuracy.append(metrics[1])  # accuracy
            
            val_bar.set_postfix({
                "val_loss": f"{metrics[0]:.4f}",
                "val_accuracy": f"{metrics[1]:.4f}"
            })
            val_bar.update(1)
        
        val_bar.close()
        
        # Summary for the epoch
        epoch_train_loss = sum(train_loss) / len(train_loss)
        epoch_train_accuracy = sum(train_accuracy) / len(train_accuracy)
        epoch_val_loss = sum(val_loss) / len(val_loss)
        epoch_val_accuracy = sum(val_accuracy) / len(val_accuracy)
        
        print(f"Epoch {epoch}: Loss={epoch_train_loss:.4f}, "
              f"Accuracy={epoch_train_accuracy:.4f}, "
              f"Val_Loss={epoch_val_loss:.4f}, "
              f"Val_Accuracy={epoch_val_accuracy:.4f}")
        
        # EarlyStopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model
            model.save(config["logging"]["checkpoint_path"])
            print(f"Model improved and saved to {config['logging']['checkpoint_path']}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
        
        if epochs_no_improve >= patience:
            print(f"Stopping early after {patience} epochs without improvement.")
            break

    print("Training completed.")

if __name__ == "__main__":
    train_model()
