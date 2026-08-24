"""Prepare detected face images for the trained models."""

import cv2
import numpy as np


IMAGE_SIZE = (224, 224)
EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]


def crop_face(frame, x, y, width, height):
    """Crop one face safely from an OpenCV frame."""
    frame_height, frame_width = frame.shape[:2]
    left = max(0, x)
    top = max(0, y)
    right = min(frame_width, x + width)
    bottom = min(frame_height, y + height)

    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def preprocess_face(face):
    """Resize a BGR face image and scale pixels for MobileNetV2."""
    if face is None or face.size == 0:
        raise ValueError("The face image is empty.")

    # OpenCV reads images as BGR, but the models were trained with RGB images.
    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    resized_face = cv2.resize(rgb_face, IMAGE_SIZE)

    # MobileNetV2 expects values between -1 and 1.
    normalized_face = resized_face.astype("float32") / 127.5 - 1

    # The model expects a batch, even when we predict one face.
    return np.expand_dims(normalized_face, axis=0)


def preprocess_face_emotion(face):
    """Resize a BGR face image to 48x48 grayscale and scale for CNN."""
    if face is None or face.size == 0:
        raise ValueError("The face image is empty.")

    # Convert to grayscale
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize to 48x48
    resized_face = cv2.resize(gray_face, (48, 48))

    # Normalize to 0-1
    normalized_face = resized_face.astype("float32") / 255.0

    # Expand dims for channel and batch: (1, 48, 48, 1)
    expanded_face = np.expand_dims(normalized_face, axis=-1)
    return np.expand_dims(expanded_face, axis=0)

