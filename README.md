# SkinLens — Setup & Run Guide

## Folder Structure

```
skinlens/
├── app.py                  ← Flask backend (entry point)
├── predict.py              ← CNN inference + questionnaire scoring
├── image_quality.py        ← Image quality validator
├── requirements.txt        ← Python dependencies
├── skinlens_model.keras    ← ⚠️  YOUR TRAINED MODEL — place here
├── uploads/                ← Temp upload folder (auto-created)
└── static/
    └── index.html          ← Frontend website
```

## Step 1 — Add your trained model

Copy your trained model file from the `cnn/` folder into this directory:

```bash
cp ../cnn/skinlens_model.keras ./skinlens_model.keras
```

The model file must be named `skinlens_model.keras` exactly (see `predict.py` → `MODEL_PATH`).

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> TensorFlow takes a few minutes. If you have a GPU: `pip install tensorflow[and-cuda]`

## Step 3 — Run

```bash
python app.py
```

Open: **http://localhost:5001**

### Production (Gunicorn)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check server + model status |
| POST | `/api/scan/quick` | Instant scan (image + form fields) |
| POST | `/api/scan/detailed` | Background scan, returns `job_id` |
| GET | `/api/scan/status/<job_id>` | Poll detailed scan result |

## Form Fields Sent by Frontend

| HTML Element ID | FormData key | predict.py key | Values |
|----------------|-------------|----------------|--------|
| q-gender | gender | gender | male/female/other |
| q-age | age | age | 0-18/19-35/36-50/51-65/65+ |
| q-sun | sun | sun | low/moderate/high |
| q-history | history | history | yes/no |
| q-dur | duration | duration | slow/rapid |
| q-change | change | change | yesd/yesr/no/unknown |
| q-existing | existing | existing | yes/no/unknown |
| q-symptoms | symptoms | itching | yes/no |
| q-bleeding | bleeding | bleeding | yes/no/unknown |
| q-oozing | oozing | oozing | yes/no/unknown |

## Risk Score Formula

```
final_risk = (0.7 × cnn_risk) + (0.3 × questionnaire_risk)

low:      final_risk < 30
moderate: 30 ≤ final_risk < 60
high:     final_risk ≥ 60
```
