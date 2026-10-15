"""Funciones de graficación reutilizadas por train/evaluate."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay


def plot_history(history: dict, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Pérdida (loss)")):
        ax.plot(history[key], label="entrenamiento")
        ax.plot(history["val_" + key], label="validación")
        ax.set_title(title); ax.set_xlabel("Época"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_confusion(y_true, y_pred, class_names, out_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax, xticks_rotation=45)
    ax.set_title("Matriz de confusión (conjunto de prueba)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_predictions(paths, y_true, probs, class_names, n=12, seed=42, out_path=None):
    """Cuadrícula de n predicciones: verde = acierto, rojo = error."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx):
        pred = int(np.argmax(probs[i]))
        ok = pred == int(y_true[i])
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.set_title(f"Real: {class_names[int(y_true[i])]}\nPred: {class_names[pred]} "
                     f"({probs[i][pred]*100:.1f}%)",
                     color="green" if ok else "red", fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
