import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def create_model(config):
    base_model = ResNet50(
        weights="imagenet" if config["model"]["pretrained"] else None,
        include_top=False,
        input_shape=tuple(config["model"]["input_shape"])
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(
        config["model"]["output_classes"], 
        activation=config["model"]["activation"]
    )(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=config["training"]["loss_function"],
        metrics=config["evaluation"]["metrics"]
    )
    
    return model
