# Arrhythmia Classification from ECG Signals

A full-stack web application that classifies cardiac arrhythmias from raw ECG signals with explainable AI output.

## Architecture

```
ecg-arrhythmia/
├── backend/                  # FastAPI Python service
│   ├── app/
│   │   ├── main.py           # FastAPI entry point
│   │   ├── api/              # Route handlers
│   │   ├── core/             # Signal processing, features, model
│   │   └── models/           # Saved ML model artifacts
│   ├── train.py              # Offline training script
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                 # Vue 3 + Vite
│   ├── src/
│   │   ├── components/       # ECG viewer, results, explanation
│   │   ├── views/            # Page views
│   │   └── api/              # Axios client
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Quickstart

### Backend

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Train model on MIT-BIH dataset (downloads automatically via wfdb)
python train.py

# Start API server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Classification Labels

| Label | Description |
|-------|-------------|
| `N`   | Normal sinus rhythm |
| `AF`  | Atrial Fibrillation |
| `PVC` | Premature Ventricular Contraction |
| `TACHY` | Tachycardia |
| `OTHER` | Other/Unknown arrhythmia |

## Pipeline

1. **Ingestion** — upload CSV/WFDB record or select a sample
2. **Pre-processing** — bandpass filter (0.5–40 Hz), baseline wander removal
3. **Segmentation** — R-peak detection via neurokit2 Pan-Tompkins
4. **Feature extraction** — HRV time-domain, beat morphology (P/Q/R/S/T amplitudes, intervals)
5. **Classification** — Random Forest (primary) or XGBoost (alt)
6. **Explainability** — SHAP force plot per beat + top-3 feature reasons
7. **LLM narration** (optional) — Ollama generates plain-language summary

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/classify` | Upload ECG, get classification |
| `GET`  | `/api/samples` | List built-in sample records |
| `GET`  | `/api/sample/{name}` | Fetch a sample record |
| `GET`  | `/health` | Health check |
