# Camera Face Recognition with OpenCV

This project provides a simple face-recognition pipeline using OpenCV's built-in LBPH recognizer.

## Features
- Capture face samples from the camera
- Train an LBPH face recognizer
- Run real-time camera recognition

## Project layout

```text
main.py   # Compatibility entrypoint
face_recognition/            # Modular Python package
  __init__.py
  camera.py                  # Camera and OpenCV helpers
  cli.py                     # Argument parsing and command flow
  config.py                  # Paths and shared configuration
  dataset.py                 # Dataset capture and preprocessing
  model.py                   # Training and model loading
  recognition.py             # Real-time recognition loop
```

## Setup

1. Create and activate a virtual environment
2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1) Capture faces for a new person

```bash
python face_camera_recognizer.py --train --name {Your Name} --samples 30
```

- A camera window will open.
- Keep your face centered so the detector can save samples.
- Press `q` to stop early.

### 2) Start recognition

```bash
python face_camera_recognizer.py --recognize
```

- If the model is missing, the app will automatically train from the saved dataset.
- The camera window will show recognized names or `Unknown`.
- Press `q` to stop.

## Notes
- The dataset is stored in the `dataset/` folder.
- The trained model is saved in the `models/` folder.
- If you add more people, run `--train` again for each person.
