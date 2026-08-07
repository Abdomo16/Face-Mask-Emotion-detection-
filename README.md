# Face Mask & Emotion Detection

This project detects if a person is wearing a face mask and what emotion they are showing,
using a webcam or uploaded image.

The project uses two Kaggle datasets:

- Face Mask Detection by `andrewmvd`
- FER2013 Facial Expression Recognition by `msambare`

## project structure

```
face-mask-emotion-detection/
├── data/raw and data/processed/  <- downloaded datasets go here (not tracked in git)
├── notebooks/
│   ├── 01_data_eda_preprocessing.ipynb  <- data loading, cleaning, EDA, class imbalance
│   ├── 02_model_training.ipynb           <- model training and evaluation
│   └── 03_merged_eda_and_training.ipynb  <- consolidated, improved end-to-end Colab workflow
├── src/                     <- Phase 3 inference modules
├── models/                  <- trained Keras weights, stored with Git LFS
├── outputs/                 <- plots and figures saved from notebooks
├── app.py                   <- streamlit app for the live demo
├── requirements.txt
└── README.md
```

## how to set up

```bash
git clone https://github.com/Abdomo16/Face-Mask-Emotion-detection-.git
cd Face-Mask-Emotion-detection-

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## training the models (Google Colab)

1. Open `notebooks/03_merged_eda_and_training.ipynb` in Google Colab.
2. Run the cells to download the datasets, perform EDA, train, evaluate, and save the models.
3. The repository includes the trained weights used for Phase 3:
   - `models/mask_model_best.h5`
   - `models/emotion_model_best.h5`

Git LFS is required when cloning model artifacts:

```bash
git lfs install
git lfs pull
```

## running the app

```bash
streamlit run app.py
```

The current app is a webcam placeholder. The end-to-end Streamlit interface and inference pipeline will be added in Phases 3 and 4.

## results

| model | task | accuracy | f1 score | roc auc |
|-------|------|----------|----------|---------|
| baseline CNN | mask | recorded in the merged notebook | - | - |
| MobileNetV2 | mask | recorded in the merged notebook | - | - |
| baseline CNN | emotion | recorded in the merged notebook | - | - |
| MobileNetV2 | emotion | recorded in the merged notebook | - | - |

## plots and outputs

the following plots are saved in the `outputs/` folder after running the notebooks:

- `plot1_class_distribution.png` - how many images per class
- `plot2_sample_grid.png` - example images for each emotion
- `plot3_size_distribution.png` - image width, height, aspect ratio
- `plot4_pixel_intensity.png` - pixel color distribution
- `plot5_class_percentages.png` - class balance as pie charts
- `plot6_average_faces.png` - average pixel face per emotion
- `plot7_correlation_heatmap.png` - pixel feature correlations
- `confusion_matrices.png` - confusion matrix for both models
- `roc_curve_mask.png` - ROC curve for mask detection
- `baseline_vs_transfer.png` - comparison of both models
- `learning_curve_mask.png` and `learning_curve_emotion.png`

## datasets

- https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- https://www.kaggle.com/datasets/msambare/fer2013
