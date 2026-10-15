"""Descarga, listado, partición y construcción de pipelines tf.data."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from . import config


def _has_images(d: Path) -> bool:
    return any(True for _ in d.glob("*.jpg")) or any(True for _ in d.glob("*.jpeg"))


def find_dataset_root(base: Path) -> Path:
    """Busca la carpeta con las subcarpetas de clase (tolera extracciones anidadas)."""
    base = Path(base)
    for cand in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        classes = [d for d in cand.iterdir() if d.is_dir() and _has_images(d)]
        if len(classes) >= 2:
            return cand
    raise RuntimeError(f"No se encontraron carpetas de clase con imágenes en {base}. "
                       "Borra la carpeta 'data' y vuelve a ejecutar.")


def download_dataset(data_dir: Path = config.DATA_DIR) -> Path:
    """Descarga y descomprime flower_photos. Devuelve la carpeta raíz del dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = tf.keras.utils.get_file(
        fname="flower_photos",
        origin=config.DATASET_URL,
        untar=True,
        cache_dir=str(data_dir),
        cache_subdir="",
    )
    extracted = Path(extracted)
    return find_dataset_root(extracted if extracted.is_dir() else data_dir)


def list_images(root: Path):
    """Devuelve (rutas, etiquetas enteras, nombres de clase) ordenados de forma determinista."""
    root = find_dataset_root(Path(root))
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir() and _has_images(d))
    paths, labels = [], []
    for idx, name in enumerate(class_names):
        files = sorted(list((root / name).glob("*.jpg")) + list((root / name).glob("*.jpeg")))
        for p in files:
            paths.append(str(p))
            labels.append(idx)
    return np.array(paths), np.array(labels, dtype="int32"), class_names


def split_data(paths, labels, val_frac=config.VAL_FRAC, test_frac=config.TEST_FRAC,
               seed=config.SEED):
    """Partición estratificada train/val/test (reproducible con la misma semilla)."""
    p_train, p_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed)
    rel_test = test_frac / (val_frac + test_frac)
    p_val, p_test, y_val, y_test = train_test_split(
        p_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed)
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE)   # float32, rango 0-255
    img.set_shape((*config.IMG_SIZE, 3))
    return img, label


def make_dataset(paths, labels, training: bool, batch_size=config.BATCH_SIZE):
    """Pipeline tf.data. La normalización (1/255) vive dentro del modelo."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
