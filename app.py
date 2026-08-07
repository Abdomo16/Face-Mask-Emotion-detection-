"""Streamlit user interface for face-mask and emotion detection."""

import cv2
import numpy as np
import streamlit as st

from src.face_detector import detect_faces, load_face_detector
from src.model import load_models, predict_emotion, predict_mask
from src.preprocessing import crop_face


st.set_page_config(
    page_title="Face Mask & Emotion Detection",
    page_icon="🙂",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the AI models...")
def load_pipeline():
    """Load the detector and models once, not every time the page refreshes."""
    detector = load_face_detector()
    mask_model, emotion_model = load_models()
    return detector, mask_model, emotion_model


def read_image(image_file):
    """Convert an uploaded or camera image into an OpenCV BGR image."""
    image_bytes = np.frombuffer(image_file.getvalue(), np.uint8)
    return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)


def analyse_image(image, detector, mask_model, emotion_model):
    """Detect faces, predict labels, and draw results on a copy of the image."""
    result_image = image.copy()
    predictions = []

    for x, y, width, height in detect_faces(image, detector):
        face = crop_face(image, x, y, width, height)
        if face is None:
            continue

        mask_label, mask_confidence = predict_mask(face, mask_model)
        emotion_label, emotion_confidence = predict_emotion(face, emotion_model)
        predictions.append(
            {
                "mask": mask_label,
                "mask_confidence": mask_confidence,
                "emotion": emotion_label,
                "emotion_confidence": emotion_confidence,
            }
        )

        color = (44, 160, 44) if mask_label == "with_mask" else (45, 75, 220)
        cv2.rectangle(result_image, (x, y), (x + width, y + height), color, 2)
        text = f"{mask_label} | {emotion_label}"
        text_y = y - 10 if y > 25 else y + height + 25
        cv2.putText(
            result_image,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return result_image, predictions


st.markdown(
    """
    <style>
        .main-title {font-size: 2.4rem; font-weight: 700; margin-bottom: 0;}
        .subtitle {color: #667085; font-size: 1.05rem; margin-top: 0.25rem;}
        .result-card {
            background: #f6f8fb; border-radius: 12px; padding: 1rem;
            border-left: 5px solid #4f46e5; margin-bottom: 0.75rem;
        }
    </style>
    <p class="main-title">Face Mask & Emotion Detection</p>
    <p class="subtitle">Upload a photo or use your camera to detect masks and facial emotions.</p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How it works")
    st.write("1. Choose an image source.")
    st.write("2. The app finds each face.")
    st.write("3. It predicts mask status and emotion.")
    st.divider()
    st.caption("Built with OpenCV, MobileNetV2, TensorFlow, and Streamlit.")

source = st.radio("Choose image source", ["Upload image", "Use camera"], horizontal=True)

if source == "Upload image":
    image_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Take a photo")

if image_file is None:
    st.info("Choose an image source above to start detection.")
    st.stop()

image = read_image(image_file)
if image is None:
    st.error("This image could not be read. Please choose another JPG or PNG file.")
    st.stop()

try:
    detector, mask_model, emotion_model = load_pipeline()
    with st.spinner("Detecting faces and making predictions..."):
        result_image, predictions = analyse_image(image, detector, mask_model, emotion_model)
except (FileNotFoundError, ImportError) as error:
    st.error(f"Setup problem: {error}")
    st.info("Run `pip install -r requirements.txt` and make sure both .h5 files are in models/.")
    st.stop()

left_column, right_column = st.columns([1.6, 1])

with left_column:
    st.subheader("Detection result")
    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)

with right_column:
    st.subheader("Predictions")
    if not predictions:
        st.warning("No face was detected. Try a clearer, front-facing photo.")
    else:
        st.success(f"Found {len(predictions)} face(s)")
        for number, prediction in enumerate(predictions, start=1):
            st.markdown(
                f"""
                <div class="result-card">
                    <b>Face {number}</b><br>
                    Mask: <b>{prediction['mask'].replace('_', ' ').title()}</b>
                    ({prediction['mask_confidence']:.0%})<br>
                    Emotion: <b>{prediction['emotion'].title()}</b>
                    ({prediction['emotion_confidence']:.0%})
                </div>
                """,
                unsafe_allow_html=True,
            )
