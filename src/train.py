"""Entrenamiento de la CNN.

Uso:  python -m src.train [--epochs 40] [--data-dir RUTA/flower_photos]
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from . import config
from .data import download_dataset, list_images, make_dataset, split_data
from .model import build_model
from .plots import plot_history


def main(epochs: int = config.EPOCHS, data_dir: str | None = None):
    tf.keras.utils.set_random_seed(config.SEED)
    for d in (config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    root = Path(data_dir) if data_dir else download_dataset()
    paths, labels, class_names = list_images(root)
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = split_data(paths, labels)
    print(f"Imágenes: {len(paths)} | train={len(p_tr)} val={len(p_va)} test={len(p_te)}")
    print("Clases:", class_names)

    train_ds = make_dataset(p_tr, y_tr, training=True)
    val_ds = make_dataset(p_va, y_va, training=False)

    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # EarlyStopping restauró los mejores pesos (menor val_loss) antes de guardar
    model.save(config.MODEL_PATH)
    config.CLASS_NAMES_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2))
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    config.HISTORY_PATH.write_text(json.dumps(hist, indent=2))
    plot_history(hist, config.FIGURES_DIR / "curvas_entrenamiento.png")
    print(f"Modelo guardado en {config.MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Carpeta flower_photos ya descargada (opcional)")
    args = ap.parse_args()
    main(args.epochs, args.data_dir)
