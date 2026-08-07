"""Detect faces with OpenCV's built-in Haar Cascade."""

import cv2


def load_face_detector():
    """Load OpenCV's ready-made frontal-face detector."""
    cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_file)

    if detector.empty():
        raise FileNotFoundError("Could not load the Haar Cascade face detector.")
    return detector


def detect_faces(frame, detector):
    """Return face boxes as (x, y, width, height)."""
    if frame is None or frame.size == 0:
        return []

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    return [tuple(face) for face in faces]
