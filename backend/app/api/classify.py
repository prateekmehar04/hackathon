"""
POST /api/classify
POST /api/classify/upload

Returns per-beat classifications, SHAP explanations, beat window positions
(for frontend waveform overlay), and an optional LLM summary.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from collections import Counter
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, field_validator
from typing import List, Optional, Literal

from app.core.signal_processing import process_signal
from app.core.features import extract_features, FEATURE_NAMES, N_FEATURES
from app.core.classifier import get_classifier
from app.core.explainer import SHAPExplainer
from app.core.llm import generate_explanation

router = APIRouter(tags=["Classify"])


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class SignalRequest(BaseModel):
    signal: List[float]
    sampling_rate: int = 360
    model_type: Literal["rf", "xgb"] = "rf"
    include_llm: bool = False
    include_shap: bool = True

    @field_validator("sampling_rate")
    @classmethod
    def check_fs(cls, v: int) -> int:
        if not (50 <= v <= 2000):
            raise ValueError("sampling_rate must be between 50 and 2000 Hz")
        return v


class FeatureImportance(BaseModel):
    name: str
    value: float
    shap: float
    direction: str


class BeatResult(BaseModel):
    beat_index: int
    rpeak_sample: int
    beat_start: int          # sample index of window start (for overlay)
    beat_end: int            # sample index of window end
    label: str
    confidence: float
    top_features: List[FeatureImportance]
    shap_values: Optional[List[float]] = None


class ClassifyResponse(BaseModel):
    filtered_signal: List[float]
    sampling_rate: int
    rpeaks: List[int]
    beats: List[BeatResult]
    dominant_label: str
    dominant_confidence: float
    label_counts: dict
    feature_names: List[str]
    llm_explanation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Shared processing helper
# --------------------------------------------------------------------------- #

def _classify_signal(
    raw: np.ndarray,
    sampling_rate: int,
    model_type: str,
    include_shap: bool,
):
    processed = process_signal(raw, sampling_rate)
    beats = processed["beats"]
    rpeaks = processed["rpeaks"]
    starts = processed["beat_starts"]
    ends = processed["beat_ends"]
    filtered = processed["filtered"]

    if len(beats) == 0:
        raise HTTPException(status_code=422, detail="No beats detected in the signal.")

    pre_samples = int(100 * sampling_rate / 360)
    X = extract_features(beats, rpeaks, sampling_rate, pre_samples=pre_samples)

    clf = get_classifier(model_type)
    labels, confidences = clf.predict_with_confidence(X)

    counter = Counter(labels)
    dominant_label = counter.most_common(1)[0][0]
    dominant_idxs = [i for i, l in enumerate(labels) if l == dominant_label]
    dominant_confidence = float(np.mean([confidences[i] for i in dominant_idxs]))

    shap_results = None
    if include_shap:
        explainer = SHAPExplainer(clf)
        pred_indices = clf.label_encoder.transform(labels).tolist()
        shap_results = explainer.explain_batch(X, pred_indices)

    beat_results = []
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        shap_data = shap_results[i] if shap_results else {"top_features": [], "shap_values": None}
        beat_results.append(
            BeatResult(
                beat_index=i,
                rpeak_sample=rpeaks[i],
                beat_start=starts[i],
                beat_end=ends[i],
                label=label,
                confidence=round(conf, 4),
                top_features=[FeatureImportance(**f) for f in shap_data.get("top_features", [])],
                shap_values=shap_data.get("shap_values"),
            )
        )

    return filtered, rpeaks, beat_results, dominant_label, dominant_confidence, counter, X


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@router.post("/classify", response_model=ClassifyResponse)
async def classify_json(body: SignalRequest):
    """Classify an ECG signal provided as a JSON array."""
    raw = np.array(body.signal, dtype=np.float32)
    filtered, rpeaks, beat_results, dominant_label, dominant_conf, counter, X = \
        _classify_signal(raw, body.sampling_rate, body.model_type, body.include_shap)

    llm_text = None
    if body.include_llm and beat_results:
        top = beat_results[0].top_features[:5]
        llm_text = await generate_explanation(
            label=beat_results[0].label,
            confidence=beat_results[0].confidence,
            top_features=[f.model_dump() for f in top],
            beat_count=len(beat_results),
            dominant_label=dominant_label,
        )

    return ClassifyResponse(
        filtered_signal=filtered.tolist(),
        sampling_rate=body.sampling_rate,
        rpeaks=rpeaks,
        beats=beat_results,
        dominant_label=dominant_label,
        dominant_confidence=round(dominant_conf, 4),
        label_counts=dict(counter),
        feature_names=FEATURE_NAMES,
        llm_explanation=llm_text,
    )


@router.post("/classify/upload", response_model=ClassifyResponse)
async def classify_upload(
    file: UploadFile = File(...),
    sampling_rate: int = Query(360, ge=50, le=2000),
    model_type: Literal["rf", "xgb"] = Query("rf"),
    include_shap: bool = Query(True),
    include_llm: bool = Query(False),
):
    """
    Classify an ECG signal from an uploaded CSV file.
    The file must contain one column of mV values (no header required).
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), header=None)
        raw = df.iloc[:, 0].to_numpy(dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    filtered, rpeaks, beat_results, dominant_label, dominant_conf, counter, X = \
        _classify_signal(raw, sampling_rate, model_type, include_shap)

    llm_text = None
    if include_llm and beat_results:
        top = beat_results[0].top_features[:5]
        llm_text = await generate_explanation(
            label=beat_results[0].label,
            confidence=beat_results[0].confidence,
            top_features=[f.model_dump() for f in top],
            beat_count=len(beat_results),
            dominant_label=dominant_label,
        )

    return ClassifyResponse(
        filtered_signal=filtered.tolist(),
        sampling_rate=sampling_rate,
        rpeaks=rpeaks,
        beats=beat_results,
        dominant_label=dominant_label,
        dominant_confidence=round(dominant_conf, 4),
        label_counts=dict(counter),
        feature_names=FEATURE_NAMES,
        llm_explanation=llm_text,
    )
