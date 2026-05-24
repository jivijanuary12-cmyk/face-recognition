from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "lbph_face_model.yml"
LABELS_PATH = MODEL_DIR / "labels.npy"


def ensure_directories() -> None:
    DATASET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
