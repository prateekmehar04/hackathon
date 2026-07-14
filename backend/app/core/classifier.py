"""
ML classification module.

Primary model : Random Forest
Alternative   : XGBoost

Both are trained on MIT-BIH Arrhythmia Database beat-level features.
After training, models are persisted to backend/app/models/.

Label mapping (AAMI standard collapsed):
  N    – Normal / Left/Right bundle branch beat / Atrial escape
  AF   – Atrial Fibrillation (approximated via HRV irregularity in beat classifier)
  PVC  – Premature Ventricular Contraction (V in MIT-BIH)
  TACHY– Tachycardia (inferred from HR > 100 bpm feature)
  OTHER– Fusion, paced, unclassifiable
"""
from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from typing import Literal, Tuple

MODEL_DIR = Path(__file__).parent.parent / "models"
RF_PATH = MODEL_DIR / "rf_model.joblib"
XGB_PATH = MODEL_DIR / "xgb_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"

# Canonical class order
CLASSES = ["N", "AF", "PVC", "TACHY", "OTHER"]


def _load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Run `python train.py` from the backend directory first."
        )
    return joblib.load(path)


class ArrhythmiaClassifier:
    """Thin wrapper that loads saved model + scaler and exposes predict/predict_proba."""

    def __init__(self, model_type: Literal["rf", "xgb"] = "rf"):
        model_path = RF_PATH if model_type == "rf" else XGB_PATH
        self.model = _load_artifact(model_path)
        self.scaler = _load_artifact(SCALER_PATH)
        self.label_encoder = _load_artifact(LABEL_ENCODER_PATH)
        self.model_type = model_type

    def predict(self, X: np.ndarray) -> list[str]:
        Xs = self.scaler.transform(X)
        indices = self.model.predict(Xs)
        return self.label_encoder.inverse_transform(indices).tolist()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability matrix (n_samples, n_classes)."""
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> Tuple[list[str], list[float]]:
        """Returns (labels, confidences) where confidence = max class probability."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        labels = self.label_encoder.inverse_transform(indices).tolist()
        confidences = proba[np.arange(len(proba)), indices].tolist()
        return labels, confidences


# Singleton — loaded once per worker process
_classifier: ArrhythmiaClassifier | None = None


def get_classifier(model_type: Literal["rf", "xgb"] = "rf") -> ArrhythmiaClassifier:
    global _classifier
    if _classifier is None or _classifier.model_type != model_type:
        _classifier = ArrhythmiaClassifier(model_type)
    return _classifier
