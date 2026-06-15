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

[🌐 Live Demo](#) · [📖 Docs](#documentation) · [🚀 Setup](#-setup-instructions) · [🤝 Team](#-team)

---

</div>

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [AI Model & Risk Algorithm](#-ai-model--risk-algorithm)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
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

1. **Computer Vision (70% Weight)** — A fine-tuned CNN analyses the uploaded lesion image for shape irregularities, border patterns, colour variation, and texture anomalies
2. **Patient Metadata (30% Weight)** — A dynamic 9-factor clinical questionnaire (age, sun exposure, family history, lesion duration, physical changes, and symptoms) accurately grounds the AI's visual findings in real-world patient context

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
│                                             (70% AI / 30% Form)    │
│                              ↓                                      │
│   🩺 Dermatologist       📄 Report           🎯 Final Verdict       │
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
| **Detailed Scan** | ~10 seconds | Multi-pass with Test-Time Augmentation (TTA) | Higher confidence results |

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
|-----------|------------|---------|
| Deep Learning Framework | TensorFlow ≥ 2.13 | Model training & inference |
| Base Architecture | MobileNetV2 (Keras) | Pretrained ImageNet backbone |
| Image Processing | Pillow ≥ 10.0 | Image loading, resizing, preprocessing |
| Numerical Computing | NumPy ≥ 1.24 | Array ops, probability computation |
| Training Utilities | scikit-learn ≥ 1.3 | Class weighting, evaluation metrics |
| Visualisation | Matplotlib ≥ 3.7 | Training history plots |

### Frontend

| Component | Technology |
|-----------|------------|
| Interface | HTML5 / CSS3 / Vanilla JS |
| Deployment Target | Any smartphone browser — no install required |
| Language Support | English · Hindi · Marathi |

---

## 🧠 AI Model & Risk Algorithm

### Architecture

SkinLens uses a **fine-tuned MobileNetV2** CNN — chosen for its balance of accuracy, inference speed (~0.3s on CPU), and small model footprint (~14MB).

```
Input (224×224×3)
      ↓
MobileNetV2 Backbone (pretrained on ImageNet)
      ↓
GlobalAveragePooling2D → BatchNormalization
      ↓
Dense(256, ReLU) → Dropout(0.4)
      ↓
Dense(128, ReLU) → Dropout(0.3)
      ↓
Dense(6, Softmax)  ← 6 output classes
```

### Output Classes

| ID | Class | Risk Tier | Colour |
|----|-------|-----------|--------|
| 0 | Benign Nevus | 🟢 Low | `#c8f560` |
| 1 | Seborrheic Keratosis | 🟢 Low | `#c8f560` |
| 2 | Dysplastic Nevus | 🟡 Moderate | `#f5a623` |
| 3 | Basal Cell Carcinoma (BCC) | 🟡 Moderate | `#f5a623` |
| 4 | Melanoma | 🔴 High | `#ff6b6b` |
| 5 | Squamous Cell Carcinoma (SCC) | 🔴 High | `#ff6b6b` |

### Dataset

Trained on the **[ISIC (International Skin Imaging Collaboration)](https://www.isic-archive.com/)** dataset — the gold standard open-access skin lesion dataset used in academic dermatology AI research.

### Training Strategy

The model is trained in two phases to maximise performance while avoiding overfitting:

**Phase 1 — Head Training (10 epochs)**
- MobileNetV2 backbone frozen
- Only the custom classification head is trained
- Allows the new head to stabilise before touching backbone weights

**Phase 2 — Fine-Tuning (20 epochs)**
- Top 30 layers of MobileNetV2 unfrozen
- Learning rate reduced 10× (`1e-5`) to avoid destroying pretrained features
- Class weights applied to handle dataset imbalance

**Data Augmentation**
```python
rotation_range=30, width_shift=0.15, height_shift=0.15,
shear_range=0.1, zoom_range=0.2,
horizontal_flip=True, vertical_flip=True,
brightness_range=[0.8, 1.2]
```

**Test-Time Augmentation (TTA)**  
For detailed scans, inference runs over 4 variants (original + horizontal flip + vertical flip + both) and averages the results — improving reliability without retraining.

### The 9-Factor Questionnaire Scoring Algorithm

Visual AI is imperfect without clinical context. SkinLens evaluates 9 specific patient parameters to generate a **Questionnaire Score** (out of 120, capped at 100). Points are mathematically weighted based on established dermatological risk factors:

| Risk Factor | Trigger Condition | Weight Penalty |
|-------------|-------------------|---------------|
| Age | 36–50 / 51–65 / 65+ | +5 / +10 / +15 |
| Sun Exposure | Moderate / High | +5 / +10 |
| Family History | Yes | +15 |
| Duration | Slow change / Rapid change | +5 / +15 |
| Lesion Changing | Yes (developing) / Unknown | +20 / +5 |
| Existing Mole | Yes / Unknown | +10 / +5 |
| Itching | Yes | +10 |
| Bleeding | Yes / Unknown | +20 / +5 |
| Oozing | Yes / Unknown | +15 / +5 |

### The Final Risk Calculation

The final risk tier is determined using a strict **70/30 weighted formula** — ensuring visual evidence leads the assessment, while severe clinical symptoms can still cross the threshold independently:

```
Final Risk Score = (0.7 × CNN Score) + (0.3 × Questionnaire Score)

🟢 Low Risk      →  Final Score < 30
🟡 Moderate Risk →  Final Score 30–59
🔴 High Risk     →  Final Score ≥ 60
```

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

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+ (developed on 3.12.7)
- pip
- 4GB+ RAM recommended for TensorFlow
- GPU optional but significantly speeds up training

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/skinlens.git
cd skinlens
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow installation can take a few minutes. For GPU support, replace `tensorflow` with `tensorflow-gpu` in `requirements.txt` and ensure CUDA is configured.

### 4. Prepare the Dataset (for training)

Download the [ISIC dataset](https://www.isic-archive.com/) and organise it as follows:

```
data/
  train/
    benign_nevus/           ← ~1000+ images per class recommended
    seborrheic_keratosis/
    dysplastic_nevus/
    bcc/
    melanoma/
    scc/
  val/
    (same structure, ~20% split)
```

### 5. Train the Model

```bash
python train_model.py
```

This will:
- Train in two phases (head → fine-tune)
- Save the best checkpoint to `model/skin_lens_cnn.h5`
- Output a classification report and training plot to `model/training_history.png`

### 6. Verify the Model

```bash
python check_model.py
```

Expected output:
```
File Size: XX.XX MB
✅ Model loaded successfully!
Model Input Shape: (None, 224, 224, 3)
Weights Checksum: XXXX.XXXX
```

### 7. Run the Application

```bash
python app.py
```

The server starts at `http://localhost:5001`

**For production deployment:**

```bash
gunicorn app:app --workers 1 --timeout 120
```

> **Cloud deployment note:** Restrict Gunicorn to a single worker (`--workers 1`) and extend the timeout to prevent memory-limit crashes on free-tier servers (Render, Heroku).

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

### `POST /api/scan/detailed`

Queues a background detailed scan with TTA. Returns a `job_id` to poll.

**Response `202`:**
```json
{
  "job_id": "a3f9e12bc0d84f...",
  "status": "queued",
  "quality": { "score": 88, "passed": true }
}
```

---

### `GET /api/scan/status/<job_id>`

Poll for the result of a detailed scan job.

**Response — while processing:**
```json
{ "status": "processing" }
```

**Response — on completion:**
```json
{
  "status": "complete",
  "result": { ... },
  "completed_at": "2026-06-10T12:34:56"
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
- [x] MobileNetV2 CNN training pipeline (two-phase fine-tuning)
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
