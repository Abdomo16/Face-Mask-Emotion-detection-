# Face Mask & Emotion Detection

A student-friendly computer-vision project that detects whether a face is wearing a mask and predicts its facial emotion. The interface is a local Streamlit website that works with an uploaded image or a photo taken from the browser camera.

## What the project does

```text
Image or camera photo
        ↓
OpenCV Haar Cascade finds each face
        ↓
Mask model predicts: with mask / without mask
        ↓
Emotion model predicts: angry, happy, sad, and more
```

The project uses:

- OpenCV Haar Cascade for fast face detection
- MobileNetV2 models for mask and emotion prediction
- Streamlit for the user interface
- Google Colab notebooks for data analysis and model training

## Project structure

```text
Face-Mask-Emotion-detection-
├── app.py                         # Streamlit website
├── data/                          # local datasets; not stored in Git
├── models/                        # trained Keras models; stored with Git LFS
├── notebooks/
│   ├── 01_data_eda_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_merged_eda_and_training.ipynb
├── src/
│   ├── face_detector.py           # detects faces
│   ├── preprocessing.py           # prepares face images for the models
│   └── model.py                   # loads models and makes predictions
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abdomo16/Face-Mask-Emotion-detection-.git
cd Face-Mask-Emotion-detection-
```

### 2. Download the trained models

The two `.h5` models are stored using Git LFS. Install Git LFS once, then download the model files:

```bash
git lfs install
git lfs pull
```

You should now have these files:

```text
models/mask_model_best.h5
models/emotion_model_best.h5
```

### 3. Create a virtual environment and install packages

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the website

```bash
streamlit run app.py
```

Streamlit will show a local link, usually `http://localhost:8501`. Open it in a browser.

### Test without a webcam

You do not need a webcam. Choose **Upload image** and upload a clear JPG or PNG face photo. The website will show:

- a box around every detected face
- mask status and confidence
- emotion and confidence

The **Use camera** option takes one photo using the browser camera. It is not continuous live video; this keeps the project easy to run and demonstrate.

## Train the models in Google Colab

Use the combined notebook for the complete training workflow:

[Open the merged EDA and training notebook in Google Colab](https://colab.research.google.com/github/Abdomo16/Face-Mask-Emotion-detection-/blob/phase-2-notebooks/notebooks/03_merged_eda_and_training.ipynb)

The notebook:

1. Downloads the datasets from Kaggle.
2. Cleans the data and explores it with charts.
3. Trains mask and emotion models.
4. Evaluates the models with metrics and plots.
5. Saves the trained Keras model files.

To run it yourself, create a Kaggle API token in your Kaggle account and follow the notebook instructions.

## Datasets

- [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [FER2013 Facial Expression Recognition](https://www.kaggle.com/datasets/msambare/fer2013)

## Limitations and future improvements

- Haar Cascade works best with clear, front-facing faces and good lighting.
- The application currently processes uploaded images or camera photos, not continuous video.
- A future version can add real-time video inference and automated tests.

## Student presentation summary

> The application first detects a face with OpenCV. Then it prepares the face image for two trained MobileNetV2 models. One model predicts whether the person wears a mask, while the second predicts the emotion. Finally, Streamlit shows the results in a simple website.
