import cv2


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    if hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return cap

    raise RuntimeError(f"Cannot open camera index {camera_index}.")


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return detector


def should_stop(window_name: str, key: int) -> bool:
    if key in (ord("q"), 27):
        return True

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def release_camera(cap: cv2.VideoCapture) -> None:
    cap.release()


def close_windows() -> None:
    cv2.destroyAllWindows()
