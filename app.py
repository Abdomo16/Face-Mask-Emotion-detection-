import streamlit as st
import cv2
import numpy as np
# from src.face_detector import detect_faces
# from src.model import predict_mask, predict_emotion

st.title("Face Mask & Emotion Detection")
st.write("This app detects face masks and emotions from a webcam feed.")

# Placeholder for real-time webcam inference
run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    if frame is None:
        st.error("Cannot read webcam")
        break
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Detect faces
    # faces = detect_faces(frame)
    # 2. For each face:
    #      crop -> predict_mask(crop) -> predict_emotion(crop)
    #      draw bounding box and text
    
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
    camera.release()
