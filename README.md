# Brain Tumor Segmentation

Deep learning-based brain tumor segmentation from MRI images using UNet with EfficientNet-B4 encoder.

---

# Live Demo

Hugging Face Space:

https://huggingface.co/spaces/Ngdkhoa/Brain_tumor_segmentation

---

# Project Overview

This project focuses on automatic brain tumor segmentation from MRI scans using semantic segmentation techniques.

The model predicts tumor regions directly from MRI images and visualizes the output as:

- Tumor overlay
- Binary segmentation mask

The application is deployed with Gradio on Hugging Face Spaces.

---

# Model Architecture

- Architecture: UNet
- Encoder: EfficientNet-B4
- Framework: PyTorch
- Library: segmentation-models-pytorch

---

# Training Pipeline

## Dataset

The dataset contains:

- Brain MRI images
- Binary tumor masks

The dataset was divided into:

- Training set
- Validation set
- Test set

---

## Data Augmentation

Albumentations was used for preprocessing and augmentation:

- Resize
- Horizontal Flip
- Vertical Flip
- Rotation
- Brightness & Contrast
- Normalization

---

# Loss Function

The model was trained using a weighted combination of:

- Tversky Loss
- BCEWithLogits Loss

This combination helps handle:

- Class imbalance
- False negatives
- Small tumor regions

---

# Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Architecture | UNet |
| Encoder | EfficientNet-B4 |
| Input Size | 256 × 256 |

---

# Threshold Optimization

After training, threshold tuning was performed on the validation set.

The threshold was selected by:

- Maximizing Recall
- Constraining Precision ≥ 0.80

This strategy helps reduce false negatives while maintaining stable segmentation quality.

---

# Evaluation Metrics

The model was evaluated using:

- Dice Score
- Recall
- Precision
- F2 Score

## Final Test Results

| Metric | Score |
|---|---|
| Dice | 0.8301 |
| Recall | 0.8809 |
| Precision | 0.8070 |
| F2 Score | 0.8555 |

---

# Demo Application

The Gradio application allows users to:

- Upload MRI images
- Predict tumor segmentation masks
- Visualize tumor overlays

The demo generates:

1. Overlay prediction
2. Binary segmentation mask

---

# Project Structure

```text
Brain_tumor_segmentation/
│
├── app.py
├── best.pth
├── requirements.txt
├── README.md
│
└── examples/
    ├── example1.jpg
    └── example2.jpg
