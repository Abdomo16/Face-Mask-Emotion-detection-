"""Load the trained models and make predictions."""

from pathlib import Path

import numpy as np

from .preprocessing import EMOTION_LABELS, preprocess_face


def load_models(
    mask_model_path="models/mask_model_best.h5",
    emotion_model_path="models/emotion_model_best.h5",
):
    """Load and return the mask model and emotion model."""
    if not Path(mask_model_path).is_file() or not Path(emotion_model_path).is_file():
        raise FileNotFoundError("Put both trained .h5 files in the models folder.")

    # Import here so the rest of the project can be read without TensorFlow loaded.
    from tensorflow.keras.models import load_model

    mask_model = load_model(mask_model_path, compile=False)
    emotion_model = load_model(emotion_model_path, compile=False)
    return mask_model, emotion_model


def predict_mask(face, mask_model):
    """Return the mask label and confidence for one BGR face crop."""
    image = preprocess_face(face)
    with_mask_probability = float(mask_model.predict(image, verbose=0)[0][0])

    if with_mask_probability >= 0.5:
        return "with_mask", with_mask_probability
    return "without_mask", 1 - with_mask_probability


def predict_emotion(face, emotion_model):
    """Return the emotion label and confidence for one BGR face crop."""
    image = preprocess_face(face)
    probabilities = emotion_model.predict(image, verbose=0)[0]
    emotion_index = int(np.argmax(probabilities))
    return EMOTION_LABELS[emotion_index], float(probabilities[emotion_index])
