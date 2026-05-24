import cv2

from face_recognition.camera import close_windows, get_face_detector, open_camera, release_camera, should_stop
from face_recognition.model import load_recognizer


def recognize_faces(camera_index: int = 0, confidence_threshold: int = 65) -> None:
    detector = get_face_detector()
    recognizer, names = load_recognizer()
    cap = open_camera(camera_index)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Unable to read frame from camera.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            for x, y, w, h in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (160, 160))
                label_id, confidence = recognizer.predict(face_roi)

                if confidence <= confidence_threshold:
                    name = names[label_id]
                    color = (0, 255, 0)
                    status = f"{name} ({confidence:.1f})"
                else:
                    color = (0, 0, 255)
                    status = f"Unknown ({confidence:.1f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(
                frame,
                "Press q or ESC to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Face recognition camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if should_stop("Face recognition camera", key):
                break
    finally:
        release_camera(cap)
        close_windows()
