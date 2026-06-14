# SkinLens CNN Module

## Overview

This module focuses on the development of the Computer Vision component of SkinLens. The objective is to analyze skin lesion images and assist in preliminary risk assessment by identifying patterns commonly associated with melanoma and benign lesions.

The CNN model serves as one component of the broader SkinLens platform and will later be combined with questionnaire-based insights to provide a more context-aware risk assessment.

---

## Objective

Develop a Convolutional Neural Network (CNN) capable of classifying skin lesion images into:

* Benign
* Melanoma

The model is intended for educational and awareness purposes and is not designed to replace professional medical diagnosis.

---

## Dataset

### Dataset Used

HAM10000 (Human Against Machine with 10000 Training Images)

### Dataset Characteristics

* Dermatoscopic skin lesion images
* Multiple lesion categories
* Associated metadata including diagnosis, age, sex, and lesion location

### Current Classification Strategy

Binary Classification:

| Class    | Label |
| -------- | ----- |
| Benign   | 0     |
| Melanoma | 1     |

---

## Planned Development Pipeline

### Phase 1: Data Loading

* Load metadata
* Verify image paths
* Analyze class distribution

### Phase 2: Data Preprocessing

* Image resizing
* Normalization
* Label mapping
* Train-validation-test split

### Phase 3: Model Development

* Design CNN architecture
* Train model on HAM10000 dataset
* Optimize performance

### Phase 4: Model Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Phase 5: Prediction Pipeline

* Accept uploaded image
* Generate prediction score
* Calculate confidence level

### Phase 6: Questionnaire Integration

* Combine image analysis with user responses
* Generate personalized risk assessment
* Improve explainability and user trust

---

## Input Features

The CNN will learn visual patterns such as:

* Shape irregularity
* Border irregularity
* Color variation
* Texture characteristics

---

## Expected Output

The model will provide:

* Predicted class

  * Benign
  * Melanoma

* Confidence score

* Risk assessment support for the SkinLens platform

---

## Project Structure

```text
cnn/
├── README.md
├── dataset_info.md
├── train.py
├── predict.py
├── data_loader.py
├── preprocess.py
└── requirements.txt
```

---

## Current Status

Project Setup Phase

Completed:

* GitHub workflow setup
* Development environment setup
* TensorFlow installation
* Dataset preparation
* Initial project structure

Upcoming:

* Metadata analysis
* Label verification
* Image preprocessing pipeline
* CNN training pipeline

---

## Future Enhancements

* Improved dataset diversity
* Better representation of Indian skin tones
* Advanced CNN architectures
* Explainable AI techniques
* Enhanced questionnaire integration
* Web application deployment
