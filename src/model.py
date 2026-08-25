"""Load the trained models and make predictions."""

from pathlib import Path

import numpy as np

from .preprocessing import EMOTION_LABELS, preprocess_face, preprocess_face_emotion


def load_models(
    mask_model_path="models/mask_model_best.h5",
    emotion_model_path="models/emotion_cnn_fixed.keras",
):
    """Load and return the mask model and emotion model."""
    if not Path(mask_model_path).is_file() or not Path(emotion_model_path).is_file():
        raise FileNotFoundError("Put the mask .h5 file and emotion .keras file in the models folder.")

    # Import here so the rest of the project can be read without TensorFlow loaded.
    from tensorflow.keras.models import load_model

    mask_model = load_model(mask_model_path, compile=False)
    emotion_model = load_model(emotion_model_path, compile=False)
    return mask_model, emotion_model


def predict_mask(face, mask_model):
    """Return the mask label and confidence for one BGR face crop."""
    image = preprocess_face(face)
    # Training used flow_from_directory, which sorts classes alphabetically:
    # {'with_mask': 0, 'without_mask': 1}. With class_mode='binary' the sigmoid
    # output is therefore the probability of class 1 = 'without_mask'.
    without_mask_probability = float(mask_model.predict(image, verbose=0)[0][0])

    if without_mask_probability >= 0.5:
        return "without_mask", without_mask_probability
    return "with_mask", 1 - without_mask_probability


def predict_emotion(face, emotion_model):
    """Return the emotion label and confidence for one BGR face crop."""
    image = preprocess_face_emotion(face)
    probabilities = emotion_model.predict(image, verbose=0)[0]
    emotion_index = int(np.argmax(probabilities))
    return EMOTION_LABELS[emotion_index], float(probabilities[emotion_index])
