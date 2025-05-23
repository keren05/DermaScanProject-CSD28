﻿# DermaScanProject-CSD28
# DermaScan AI: Dermatological Image Classification and Clinical Decision Support

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)

## 🌐 Project Overview

**DermaScan AI** is an intelligent dermatology assistant that classifies common skin conditions from images and provides personalized, context-aware treatment recommendations. The system integrates computer vision, clinical heuristics (e.g., ABCDE criteria), skin type data, and regional climate to support dermatological assessment — especially in low-resource settings.

---
#Project Structure
DermaScan/
├── app.py                # Streamlit app with image upload + chatbot
├── train.py              # Model training script using EfficientNetB0
├── model/                # Saved trained models
├── data/                 # ISIC 2019 images or dataset path
├── utils/                # Helper functions (e.g., preprocessing, predictions)
├── docs/                 # Architecture diagrams & documentation
└── requirements.txt      # Python dependencies

## 🎯 Objectives

- ✅ Achieve >85% validation accuracy using EfficientNetB0.
- 🧮 Address class imbalance with weighted loss functions.
- 🌍 Provide climate- and skin-type-specific treatment suggestions.
- 📱 Deploy lightweight FP16-quantized model for mobile and edge devices.
- 🧠 Enhance interpretability with confidence scores and GPT-generated explanations.
- 🧑‍⚕️ Align with ABCDE/ISIC dermatology guidelines.
- 🌐 Deliver a multilingual, intuitive Streamlit-based UI.
- 🔁 Support telemedicine integration via API endpoints.
- 🧪 Validate performance on real-world patient image samples.

---

## 🏗️ System Architecture

- **Image Preprocessing Pipeline**: Resizing, normalization, augmentation
- **Classifier**: Fine-tuned EfficientNetB0 using transfer learning
- **Clinical Decision Engine**:
  - ABCDE melanoma heuristics
  - Fitzpatrick skin type and UV exposure logic
- **Streamlit Web UI**: For patient input, prediction, chatbot explanations

> ![System Design](./docs/DermaScan_System_Design.png)

---

## 🧠 Model Training

- Backbone: `EfficientNetB0` (ImageNet weights)
- Optimizer: `Adam`, Learning Rate = `0.001`
- Loss: `Categorical Crossentropy` with `class weights`
- Augmentation: Rotation, flip, zoom
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`

```python
train_datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.1,
    rescale=1./255
)

