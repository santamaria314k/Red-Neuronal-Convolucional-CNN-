"""Configuración central del proyecto (una sola fuente de verdad)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "flower_cnn.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PATH = REPORTS_DIR / "history.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Dataset oficial de TensorFlow (fotos de flores, 5 clases)
DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/flower_photos.tgz"
)

# Hiperparámetros
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42
LEARNING_RATE = 1e-3

# Partición estratificada 70 / 15 / 15
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Nombres legibles para la interfaz
CLASS_LABELS_ES = {
    "daisy": "Margarita",
    "dandelion": "Diente de león",
    "roses": "Rosa",
    "sunflowers": "Girasol",
    "tulips": "Tulipán",
}

# Por debajo de esta confianza la app avisa que la predicción es incierta
CONFIDENCE_THRESHOLD = 0.60
