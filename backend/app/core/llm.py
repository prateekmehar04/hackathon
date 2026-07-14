"""
Ollama LLM integration for plain-language explanation generation.

Calls the local Ollama REST API (default: http://localhost:11434).
Falls back gracefully if Ollama is not running.

Usage
-----
    from app.core.llm import generate_explanation
    text = await generate_explanation(label="PVC", confidence=0.91, top_features=[...])
"""
from __future__ import annotations

import httpx
import json
from typing import List

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
TIMEOUT_SECONDS = 30


def _build_prompt(
    label: str,
    confidence: float,
    top_features: List[dict],
    beat_count: int,
    dominant_label: str,
) -> str:
    feature_lines = "\n".join(
        f"  - {f['name']}: {f['value']:.3f} (SHAP={f['shap']:+.3f}, {f['direction']})"
        for f in top_features
    )
    return f"""You are a clinical-assistant AI. Explain the following ECG classification result
in plain language suitable for a medical professional reviewing an automated arrhythmia report.
Be concise (3–5 sentences). Do NOT give treatment advice.

Classification result:
  Beat class       : {label}
  Confidence       : {confidence * 100:.1f}%
  Total beats      : {beat_count}
  Dominant rhythm  : {dominant_label}

Top contributing features:
{feature_lines}

Explanation:"""


async def generate_explanation(
    label: str,
    confidence: float,
    top_features: List[dict],
    beat_count: int = 1,
    dominant_label: str = "",
    model: str = OLLAMA_MODEL,
) -> str | None:
    """
    Call Ollama to generate a plain-language explanation.
    Returns None if Ollama is unavailable.
    """
    prompt = _build_prompt(label, confidence, top_features, beat_count, dominant_label or label)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 200},
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception:
        return None
