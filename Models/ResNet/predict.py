import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import load_config

def predict_images():
    config = load_config()
    model = tf.keras.models.load_model(config["logging"]["checkpoint_path"])
    
    pred_path = config["dataset"]["pred_data_path"]
    input_size = tuple(config["model"]["input_shape"][:2])
    image_files = os.listdir(pred_path)
    
    print("Running Predictions...")
    progress_bar = tqdm(total=len(image_files), desc="Predicting", unit="image")
    
    for img_file in image_files:
        img_path = os.path.join(pred_path, img_file)
        img = load_img(img_path, target_size=input_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction)
        class_label = config["dataset"]["class_labels"][class_idx]
        
        print(f"Image: {img_file}, Predicted Class: {class_label}")
        progress_bar.update(1)
    
    progress_bar.close()

if __name__ == "__main__":
    predict_images()
