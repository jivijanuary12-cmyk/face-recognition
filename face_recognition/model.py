import numpy as np

import cv2

from face_recognition.config import LABELS_PATH, MODEL_PATH, ensure_directories
from face_recognition.dataset import get_person_directories, prepare_training_data


def train_recognizer() -> None:
    ensure_directories()
    faces, labels = prepare_training_data()
    names = [folder.name for folder in get_person_directories()]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.write(str(MODEL_PATH))
    np.save(str(LABELS_PATH), np.asarray(names, dtype=object))


def load_recognizer() -> tuple[cv2.face_LBPHFaceRecognizer, list[str]]:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("No trained model found. Run training first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    names = np.load(str(LABELS_PATH), allow_pickle=True).tolist()
    return recognizer, names
