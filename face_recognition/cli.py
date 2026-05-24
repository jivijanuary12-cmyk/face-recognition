import argparse

from face_recognition.dataset import collect_training_data
from face_recognition.model import train_recognizer
from face_recognition.recognition import recognize_faces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenCV camera face recognizer")
    parser.add_argument("--train", action="store_true", help="Train the recognizer using captured faces")
    parser.add_argument("--name", type=str, help="Name to save when using --train")
    parser.add_argument("--samples", type=int, default=30, help="Number of face samples to capture per person")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use")
    parser.add_argument("--recognize", action="store_true", help="Run real-time recognition")
    parser.add_argument("--confidence", type=int, default=65, help="Maximum confidence value to accept a recognized face")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.train:
        if not args.name:
            raise SystemExit("Please provide --name when using --train.")
        collect_training_data(args.name, samples_to_capture=args.samples, camera_index=args.camera_index)
        train_recognizer()
        print(f"Training complete for {args.name}.")
        return

    if args.recognize:
        recognize_faces(camera_index=args.camera_index, confidence_threshold=args.confidence)
        return

    print("No action selected. Use --train to add a new person or --recognize to start the camera.")


if __name__ == "__main__":
    main()
