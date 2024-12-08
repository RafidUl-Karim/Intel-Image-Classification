import tensorflow as tf

# Check if TensorFlow is detecting the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print(f"- {gpu}")
else:
    print("No GPU detected, TensorFlow is using CPU.")


import tensorflow as tf

tensor = tf.constant([])
print(tensor.device)