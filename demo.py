import os
import random
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('/Users/rafid/OneDrive/Desktop/VSCode/Python/Image Classification/Saved Models/Keras-CNN-EPOCHS_100test_acc_0.802.h5')

# Define labels and images (modify based on your dataset classes)
labels = ['Building', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
# Path to image directories
image_dirs = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Get a list of all images in the directories
image_paths = []
for dir_name in image_dirs:
    dir_path = os.path.join('/Users/rafid/OneDrive/Desktop/VSCode/Python/Image Classification/Datasets/Intel Image Classification/seg_test/seg_test', dir_name)
    image_paths.extend([os.path.join(dir_path, img) for img in os.listdir(dir_path) if img.endswith(('jpg', 'jpeg', 'png'))])

# Randomly select 9 images
random_images = random.sample(image_paths, 9)

st.title("Intel Image Classification")
st.write("Choose an image, and the model will predict the category.")

# Display 9 images as selectable options
col1, col2, col3 = st.columns(3)
selected_image = None

with col1:
    st.image(random_images[0], use_column_width=True, caption="Image 1")
    if st.button("Select", key="img1"):
        selected_image = random_images[0]
with col2:
    st.image(random_images[1], use_column_width=True, caption="Image 2")
    if st.button("Select", key="img2"):
        selected_image = random_images[1]
with col3:
    st.image(random_images[2], use_column_width=True, caption="Image 3")
    if st.button("Select", key="img3"):
        selected_image = random_images[2]

with col1:
    st.image(random_images[3], use_column_width=True, caption="Image 4")
    if st.button("Select", key="img4"):
        selected_image = random_images[3]
with col2:
    st.image(random_images[4], use_column_width=True, caption="Image 5")
    if st.button("Select", key="img5"):
        selected_image = random_images[4]
with col3:
    st.image(random_images[5], use_column_width=True, caption="Image 6")
    if st.button("Select", key="img6"):
        selected_image = random_images[5]

with col1:
    st.image(random_images[6], use_column_width=True, caption="Image 7")
    if st.button("Select", key="img7"):
        selected_image = random_images[6]
with col2:
    st.image(random_images[7], use_column_width=True, caption="Image 8")
    if st.button("Select", key="img8"):
        selected_image = random_images[7]
with col3:
    st.image(random_images[8], use_column_width=True, caption="Image 9")
    if st.button("Select", key="img9"):
        selected_image = random_images[8]

# If an image is selected, process and predict
if selected_image:
    st.image(selected_image, caption="Selected Image", use_column_width=True)
    
    # Open the image using PIL
    image = Image.open(selected_image)
    
    # Resize image to the target size
    image = image.resize((150, 150))

    # Convert image to a numpy array
    image = np.array(image)

    # Normalize the image
    image = image / 255.0

    # Expand dimensions to match the model input
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class} ({confidence * 100:.2f}%)")
