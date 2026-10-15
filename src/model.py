"""Definición de la CNN (normalización y data augmentation incluidos en el modelo)."""
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_augmentation() -> keras.Sequential:
    """Capas de aumento de datos: solo se activan durante el entrenamiento."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.MaxPooling2D()(x)


def build_model(num_classes: int, img_size=config.IMG_SIZE) -> keras.Model:
    inputs = keras.Input(shape=(*img_size, 3), name="imagen")   # píxeles 0-255
    x = layers.Rescaling(1.0 / 255, name="normalizacion")(inputs)
    x = build_augmentation()(x)
    for filters in (32, 64, 128, 256):
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilidades")(x)
    model = keras.Model(inputs, outputs, name="cnn_flores")
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
