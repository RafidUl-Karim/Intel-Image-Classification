import tensorflow as tf

def prepare_datasets(config):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=config["training"]["augmentation"]["horizontal_flip"],
        rotation_range=config["training"]["augmentation"]["rotation_range"],
        zoom_range=config["training"]["augmentation"]["zoom_range"]
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    
    train_generator = train_datagen.flow_from_directory(
        config["dataset"]["train_data_path"],
        target_size=config["model"]["input_shape"][:2],
        batch_size=config["training"]["batch_size"],
        class_mode="categorical"
    )
    
    test_generator = test_datagen.flow_from_directory(
        config["dataset"]["test_data_path"],
        target_size=config["model"]["input_shape"][:2],
        batch_size=config["training"]["batch_size"],
        class_mode="categorical"
    )
    
    return train_generator, test_generator
