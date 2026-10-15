"""Evaluación del modelo entrenado sobre el conjunto de PRUEBA (nunca visto).

Uso:  python -m src.evaluate [--data-dir RUTA/flower_photos]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow import keras

from . import config
from .data import download_dataset, list_images, make_dataset, split_data
from .plots import plot_confusion, plot_predictions


def main(data_dir: str | None = None):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(data_dir) if data_dir else download_dataset()
    paths, labels, class_names = list_images(root)
    saved = json.loads(config.CLASS_NAMES_PATH.read_text())
    assert saved == class_names, "Las clases del dataset no coinciden con las del modelo"
    _, _, (p_te, y_te) = split_data(paths, labels)   # misma semilla => mismo test

    model = keras.models.load_model(config.MODEL_PATH)
    test_ds = make_dataset(p_te, y_te, training=False)   # sin shuffle: orden = y_te
    loss, acc = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    report = classification_report(y_te, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "n_test": int(len(y_te)),
        "classification_report": report,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    plot_confusion(y_te, y_pred, class_names, config.FIGURES_DIR / "matriz_confusion.png")
    plot_predictions(p_te, y_te, probs, class_names, n=12,
                     out_path=config.FIGURES_DIR / "predicciones_12.png")
    print(f"Test accuracy: {acc:.4f} | macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None)
    main(ap.parse_args().data_dir)
