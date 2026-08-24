"""Streamlit user interface for live face-mask and emotion detection."""

import time

import cv2
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


def analyse_frame(image, detector, mask_model, emotion_model):
    """Detect faces in one frame, predict labels, and draw results on a copy."""
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
    <p class="subtitle">Live webcam detection of masks and facial emotions, in real time.</p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How it works")
    st.write("1. Press **Start camera**.")
    st.write("2. The app finds each face in the live video.")
    st.write("3. It predicts mask status and emotion on every frame.")
    st.divider()
    st.caption("Built with OpenCV, MobileNetV2, TensorFlow, and Streamlit.")

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

controls = st.columns(2)
start_button = controls[0].button(
    "▶ Start camera", type="primary", disabled=st.session_state.camera_on
)
stop_button = controls[1].button("⏹ Stop camera", disabled=not st.session_state.camera_on)

if start_button:
    st.session_state.camera_on = True
    st.rerun()
if stop_button:
    st.session_state.camera_on = False

frame_slot = st.empty()
status_slot = st.empty()

if not st.session_state.camera_on:
    frame_slot.info("Press **Start camera** to begin live detection from your webcam.")
    st.stop()

try:
    detector, mask_model, emotion_model = load_pipeline()
except (FileNotFoundError, ImportError) as error:
    st.session_state.camera_on = False
    st.error(f"Setup problem: {error}")
    st.info("Run `pip install -r requirements.txt` and make sure the .h5 and .keras files are in models/.")
    st.stop()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    st.session_state.camera_on = False
    st.error("Could not open your camera. Close other apps using it and try again.")
    st.stop()

frame_count = 0
fps = 0.0
start_time = time.time()

try:
    while st.session_state.camera_on:
        grabbed, frame = camera.read()
        if not grabbed:
            status_slot.error("The camera frame could not be read. Stopping live detection.")
            break

        result_image, predictions = analyse_frame(frame, detector, mask_model, emotion_model)

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        frame_slot.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        if predictions:
            lines = [f"**{len(predictions)} face(s) detected** — {fps:.1f} FPS"]
            for number, prediction in enumerate(predictions, start=1):
                lines.append(
                    f"- **Face {number}**: Mask *{prediction['mask'].replace('_', ' ')}* "
                    f"({prediction['mask_confidence']:.0%}) · "
                    f"Emotion *{prediction['emotion']}* ({prediction['emotion_confidence']:.0%})"
                )
            status_slot.markdown("\n".join(lines))
        else:
            status_slot.warning(f"No face detected — {fps:.1f} FPS")
finally:
    camera.release()
