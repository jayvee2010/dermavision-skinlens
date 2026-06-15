<div align="center">

# 🔬 SkinLens
### AI-Based Early Skin Risk Awareness Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-02C39A?style=for-the-badge)](https://keras.io/api/applications/mobilenet/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Build%20Phase-f5a623?style=for-the-badge)]()

**Upload. Analyse. Act.**  
*Skin cancer screening made accessible for every Indian — in the language they speak, on the phone they carry.*

[🌐 Live Demo](#-https://dermavision-skinlens.onrender.com/#) · [📖 Docs](#-technology-stack) · [🤝 Team](#-team)

---

</div>

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [AI Model & Risk Algorithm](#-ai-model--risk-algorithm)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Image Quality Standards](#-image-quality-standards)
- [Risk Classification](#-risk-classification)
- [Multilingual Support](#-multilingual-support)
- [Roadmap](#-roadmap)
- [Team](#-team)

---

## ❗ Problem Statement

Skin cancer is one of the most underdiagnosed conditions in India — not because people don't care, but because access to dermatological care is severely limited.

- **1 in 3 skin cancers** goes undetected until it has progressed to an advanced stage
- India has a critically **low patient-to-dermatologist ratio**, with specialists concentrated in metros
- **Rural and underserved communities** have virtually no affordable access to early screening
- Language barriers further limit the reach of existing digital health tools

The window for effective, low-cost treatment is the **early stage** — but most people never get screened at that point. SkinLens exists to change that.

---

## 💡 Solution Overview

SkinLens is a **mobile-first, AI-powered skin risk awareness web platform** that enables anyone with a smartphone to get a preliminary screening of a skin lesion — in seconds, for free, in their own language.

The platform combines two distinct data sources to generate a **personalised, weighted risk assessment**:

1. **Computer Vision (70% Weight)** — A fine-tuned MobileNetV2 CNN analyses the uploaded lesion image to identify visual patterns commonly associated with suspicious skin lesions, helping estimate the likelihood of melanoma-like characteristics.
2. **Patient Metadata (30% Weight)** — A dynamic 9-factor clinical questionnaire (including age, sun exposure, family history, lesion duration, physical changes, and symptoms) enhances the AI's image-based assessment by integrating real-world patient context and risk indicators into the final evaluation.


Results are mathematically combined into three risk tiers and delivered with **actionable guidance in English, Hindi, and Marathi**. High-risk results include a prompt to connect with a nearby dermatologist.

> ⚠️ **Disclaimer:** SkinLens is a screening and awareness tool only. It does not provide a medical diagnosis. All results should be validated by a qualified dermatologist.

---

## 🔄 How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   📸 Upload          📋 Questionnaire      🧠 AI Analysis          │
│   ─────────          ──────────────────    ────────────────         │
│   User captures  →   9 health factors   →  CNN analyses visual      │
│   lesion image       (history, changes,    patterns. Algorithm      │
│                       symptoms, age,        calculates weighted     │
│                       sun exposure)         combined risk score     │
│                                             (70% AI / 30% Form)     │
│                              ↓                                      │
│   🩺 Dermatologist       📄 Report           🎯 Final Verdict      │
│   ─────────────────      ──────────          ────────────           │
│   Connect to nearby  ←   Quick overview  ←   Low / Moderate /       │
│   doctors for high       or detailed          High — with           │
│   risk results           4-day deep scan      guidance in           │
│                                               your language         │
└─────────────────────────────────────────────────────────────────────┘
```

### Scan Modes

| Mode | Speed | Method | Best For |
|------|-------|--------|---------|
| **Quick Scan** | ~2 seconds | Single-pass CNN inference + metadata scoring | First-look screening |
| **Detailed Scan** (future development) | ~4 days | Multi-pass with Test-Time Augmentation (TTA) | Higher confidence results |

---

## 🛠 Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | Flask 3.1.2 | REST API routing and static file serving |
| CORS | Flask-CORS 4.0.0 | Secure cross-origin bridge for the frontend |
| Production Server | Gunicorn | Single-worker restricted deployment |
| Concurrency | Python `threading` | Background detailed scan jobs |

### AI & Machine Learning

| Component | Technology | Purpose |
|------------|------------|------------|
| Deep Learning Framework | TensorFlow / Keras | Model training and inference |
| Base Architecture | MobileNetV2 | Pretrained ImageNet backbone for skin lesion classification |
| Image Processing | Pillow (PIL) | Image loading, resizing, and preprocessing |
| Numerical Computing | NumPy | Array operations and prediction processing |
| Data Processing | Pandas | Dataset loading and preprocessing |
| Training Utilities | Scikit-learn | Train-validation split, class weighting, and evaluation |

### Frontend

| Component | Technology |
|-----------|------------|
| Interface | HTML5 / CSS3 / Vanilla JS |
| Deployment Target | Any smartphone browser — no install required |
| Language Support | English · Hindi · Marathi |

---

## 🧠 AI Model & Risk Algorithm

### Architecture

SkinLens uses a **fine-tuned MobileNetV2** convolutional neural network (CNN) trained on the HAM10000 skin lesion dataset. MobileNetV2 was selected for its lightweight architecture, fast inference, and strong transfer-learning capabilities.

```text
Input Image (224×224×3)
          ↓
MobileNetV2 Backbone
(Pretrained on ImageNet)
          ↓
GlobalAveragePooling2D
          ↓
Dropout (0.3)
          ↓
Dense (1, Sigmoid)
          ↓
Melanoma Risk Probability
```

### Classification Approach

The model performs **binary classification** by estimating the probability that an uploaded lesion exhibits **melanoma-like characteristics**.

| Output | Interpretation                                 |
| ------ | ---------------------------------------------- |
| 0      | No significant melanoma-like patterns detected |
| 1      | Melanoma-like patterns detected                |

The CNN serves as a screening component and is combined with patient-provided clinical information for a more comprehensive risk assessment.

### Dataset

The model was trained using the **HAM10000 (Human Against Machine with 10,000 Training Images)** dataset, one of the most widely used public dermatology imaging datasets.

For this project:

* **Melanoma (mel)** images were treated as the positive class.
* All other lesion categories were grouped as the negative class.
* Class weights were applied during training to address dataset imbalance and improve melanoma detection sensitivity.

### Training Strategy

The training process leveraged transfer learning using pretrained ImageNet weights.

**Feature Extraction Phase**

* MobileNetV2 initialized with pretrained ImageNet weights.
* Classification head trained on skin lesion images.
* Backbone initially frozen to retain learned visual features.

**Fine-Tuning Phase**

* MobileNetV2 layers unfrozen for further training.
* Model adapted to skin lesion characteristics present in HAM10000.
* Class weighting applied to reduce bias caused by class imbalance.

### Questionnaire-Based Clinical Assessment

Visual analysis alone cannot capture all patient-specific risk factors. SkinLens incorporates a **9-factor questionnaire** to provide additional clinical context.

The questionnaire evaluates:

* Age group
* Gender
* Daily sun exposure
* Family history of skin cancer
* Pain level
* Duration of lesion change
* Observable lesion development
* Existing mole history
* Symptoms such as itching, bleeding, and oozing

Each response contributes to a weighted risk score based on commonly recognized dermatological risk indicators.

### Final Risk Calculation

SkinLens combines image-based risk assessment with questionnaire-based clinical context using a weighted scoring approach:

```text
Final Risk Score =
(0.7 × CNN Risk Score)
+
(0.3 × Questionnaire Risk Score)
```

### Risk Categories

| Risk Level       | Final Score |
| ---------------- | ----------- |
| 🟢 Low Risk      | < 30        |
| 🟡 Moderate Risk | 30 – 59     |
| 🔴 High Risk     | ≥ 60        |

### Explainability

Rather than returning only a prediction, SkinLens provides:

* CNN-based image assessment
* Questionnaire-based risk assessment
* Combined risk score
* Risk category
* Key contributing risk factors identified from questionnaire responses

This helps users understand which factors influenced the final assessment while maintaining transparency in the decision-making process.

> **Disclaimer:** SkinLens is intended as a screening and awareness tool only. It does not provide a medical diagnosis and should not replace consultation with a qualified dermatologist.


---

## 📁 Project Structure

```
skinlens/
│
├── app.py                  # Flask application — API routes & server
├── predictor.py            # CNN inference engine (SkinLensPredictor)
├── train_model.py          # Model training script (MobileNetV2 fine-tuning)
├── image_quality.py        # Image quality validator (blur, brightness, resolution)
├── check_model.py          # Quick model integrity checker
├── requirements.txt        # Python dependencies
│
├── model/
│   └── skin_lens_cnn.h5    # Trained Keras model (generated by train_model.py)
│
├── data/                   # Training data (not committed — see Dataset section)
│   ├── train/
│   │   ├── benign_nevus/
│   │   ├── seborrheic_keratosis/
│   │   ├── dysplastic_nevus/
│   │   ├── bcc/
│   │   ├── melanoma/
│   │   └── scc/
│   └── val/
│       └── (same structure)
│
├── static/                 # Frontend assets
│   └── index.html          # Main web interface
│
└── uploads/                # Temp storage for uploaded images (auto-deleted)
```

---

## 📡 API Reference

### `POST /api/scan/quick`

Runs instant CNN inference and calculates the weighted clinical risk score.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | ✅ | JPG, PNG, or WebP — max 10MB |
| `gender` | string | ❌ | `male` / `female` / `other` |
| `age` | string | ❌ | `under-30` / `30-50` / `50-65` / `65+` |
| `sun` | string | ❌ | `low` / `moderate` / `high` |
| `history` | string | ❌ | `yes` / `no` |
| `duration` | string | ❌ | `slow` / `rapid` |
| `change` | string | ❌ | `yes` / `unknown` / `no` |
| `existing` | string | ❌ | `yes` / `no` / `unknown` |
| `itching` | string | ❌ | `yes` / `no` |
| `bleeding` | string | ❌ | `yes` / `no` / `unknown` |
| `oozing` | string | ❌ | `yes` / `no` / `unknown` |

**Response `200`:**
```json
{
  "risk_level": "moderate",
  "risk_label": "Moderate Risk",
  "top_class": "Dysplastic Nevus",
  "confidence": "74.3%",
  "cnn_score": 62.1,
  "questionnaire_score": 35,
  "final_score": 54.0,
  "classes": [
    { "name": "Dysplastic Nevus", "probability": 74.3, "color": "#f5a623" },
    { "name": "Benign Nevus",     "probability": 18.1, "color": "#c8f560" }
  ],
  "guidance": {
    "en": "This lesion shows features that warrant medical attention...",
    "hi": "इस घाव में ऐसी विशेषताएं हैं...",
    "mr": "या जखमेत वैद्यकीय लक्ष आवश्यक..."
  },
  "quality": { "score": 88, "passed": true },
  "metadata": { "age": "50-65", "sun": "high" }
}
```

**Error `422` — Image quality too low:**
```json
{
  "error": "Image quality too low",
  "quality": { "score": 30, "passed": false },
  "message": "Please upload a clearer, well-lit image of the lesion."
}
```

---


### `GET /api/health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## 🖼 Image Quality Standards

SkinLens validates every uploaded image before running inference. Poor-quality images produce unreliable results — uploads failing the quality check are rejected with specific, actionable feedback.

| Check | Threshold | Deduction if Failed |
|-------|-----------|-------------------|
| Resolution | Min 100×100px | −40 points |
| Brightness | Mean 40–220 (0–255) | −25 points |
| Sharpness (Laplacian variance) | ≥ 80 | −25 points |
| Colour Saturation | Mean channel range ≥ 10 | −10 points |

**Minimum passing score: 40 / 100**

**Tips for a good photo:**
- Hold the camera 10–15cm from the lesion
- Use natural light or a lamp — avoid harsh flash
- Keep the phone steady; tap to focus before capturing
- Ensure the lesion fills at least 30% of the frame

---

## 🎯 Risk Classification

| Level | Label | Action |
|-------|-------|--------|
| 🟢 Low | Low Risk | Likely benign. Continue self-monitoring; routine dermatology check at next visit. |
| 🟡 Moderate | Moderate Risk | Warrants attention. Consult a dermatologist within 2–4 weeks. |
| 🔴 High | High Risk | Characteristics associated with higher-risk conditions. Seek urgent evaluation within days. |

All risk levels are delivered in **English, Hindi, and Marathi** with specific next-step guidance.

---

## 🌐 Multilingual Support

SkinLens delivers all risk guidance and clinical recommendations in three languages to maximise accessibility across India:

| Language | Code | Coverage |
|----------|------|---------|
| English | `en` | Full |
| Hindi (हिन्दी) | `hi` | Full |
| Marathi (मराठी) | `mr` | Full |

---

## 🗺 Roadmap

### Current — Build Phase
- [x] Flask REST API with quick + detailed scan endpoints
- [x] 9-factor weighted questionnaire scoring algorithm
- [x] Image quality validation with per-check feedback
- [x] Trilingual guidance output (EN / HI / MR)
- [x] Privacy-by-design — auto-deletion of uploaded images after processing

### Upcoming
- [ ] **Inclusive Dataset** — Partner with Indian dermatology institutions to build a dataset better representing Indian skin tones and regional conditions
- [ ] **Advanced Image Quality** — Real-time capture guidance using the device camera API
- [ ] **Dermatologist Connect** — Location-aware referral system integrated with doctor availability
- [ ] **Context-Aware Analysis** — Further refinement of questionnaire weighting based on clinical validation
- [ ] **Community & NGO Integration** — Outreach partnerships for rural and underserved populations
- [ ] **Mobile App** — Native iOS and Android builds for offline-capable screening

---

## 👥 Team

**Team SkinLens** — Built for the Hackathon Build Phase

| Name | Role |
|------|------|
| Akansha Wadhwani | `[Role — TBD]` |
| Himanshi Thakur | `[Role — TBD]` |
| Jayvee Shah | `[Role — TBD]` |
| Kalindi Joshi | `[Role — TBD]` |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**SkinLens** · AI-Based Early Skin Risk Awareness · Built with ❤️ for India

*This tool is for screening awareness only and does not constitute a medical diagnosis.*

</div>
