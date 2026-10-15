"""Predicción sobre una imagen individual (usada por la CLI y por Streamlit).

Uso:  python -m src.predict ruta/a/imagen.jpg
"""
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras

from . import config


def load_artifacts(model_path=config.MODEL_PATH, class_names_path=config.CLASS_NAMES_PATH):
    model = keras.models.load_model(model_path)
    class_names = json.loads(open(class_names_path, encoding="utf-8").read())
    return model, class_names


def preprocess(img: Image.Image) -> np.ndarray:
    """Mismo preprocesamiento que en entrenamiento: RGB + resize bilineal, rango 0-255.
    (La división por 255 la hace la capa Rescaling dentro del modelo.)"""
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = tf.image.resize(np.asarray(img, dtype="float32"), config.IMG_SIZE).numpy()
    return arr[None, ...]


def predict_image(model, class_names, img: Image.Image) -> dict:
    probs = model.predict(preprocess(img), verbose=0)[0]
    top = int(np.argmax(probs))
    return {
        "class": class_names[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python -m src.predict ruta/a/imagen.jpg")
    model, class_names = load_artifacts()
    res = predict_image(model, class_names, Image.open(sys.argv[1]))
    print(f"Clase: {res['class']}  |  Confianza: {res['confidence']*100:.2f}%")
    for c, p in sorted(res["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<12} {p*100:6.2f}%")
