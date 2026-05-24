from pathlib import Path

import cv2
import numpy as np

from face_recognition.camera import get_face_detector, should_stop
from face_recognition.config import DATASET_DIR, ensure_directories

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FACE_SIZE = (160, 160)


def sanitize_name(name: str) -> str:
    return name.strip().replace(" ", "_") or "person"


def get_person_directories() -> list[Path]:
    return [path for path in sorted(DATASET_DIR.iterdir()) if path.is_dir()]


def get_training_samples() -> list[Path]:
    samples = []
    for person_dir in get_person_directories():
        for image_path in sorted(person_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                samples.append(image_path)
    return samples


def collect_training_data(name: str, samples_to_capture: int = 30, camera_index: int = 0) -> None:
    ensure_directories()
    person_dir = DATASET_DIR / sanitize_name(name)
    person_dir.mkdir(exist_ok=True)

    detector = get_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    saved = 0
    sample_id = len(list(person_dir.iterdir())) + 1

    try:
        while saved < samples_to_capture:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Unable to read frame from camera.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_roi = gray[y:y + h, x:x + w]
                resized = cv2.resize(face_roi, FACE_SIZE)
                output_path = person_dir / f"{sample_id:03d}.png"
                cv2.imwrite(str(output_path), resized)
                sample_id += 1
                saved += 1

            cv2.putText(
                frame,
                f"Capturing {saved}/{samples_to_capture} for {name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Face capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if should_stop("Face capture", key):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if saved == 0:
        raise RuntimeError("No faces were captured. Make sure your face is visible and try again.")


def prepare_training_data() -> tuple[np.ndarray, np.ndarray]:
    labels = []
    faces = []

    for person_idx, person_dir in enumerate(get_person_directories()):
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            faces.append(cv2.resize(image, FACE_SIZE))
            labels.append(person_idx)

    if not faces:
        raise RuntimeError(
            "No training images found. Add people by running: python face_camera_recognizer.py --train --name <name>"
        )

    return np.asarray(faces, dtype=np.uint8), np.asarray(labels, dtype=np.int32)
