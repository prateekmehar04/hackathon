"""
Sample ECG data management.

Loads a small set of demo records from PhysioNet MIT-BIH Arrhythmia Database
using wfdb. Records are cached locally in backend/data/samples/.
"""
from __future__ import annotations

import numpy as np
import wfdb
from pathlib import Path
from typing import List

SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Curated subset of MIT-BIH records with interesting arrhythmia content
SAMPLE_RECORDS = {
    "100": {"label": "Normal sinus rhythm", "db": "mitdb"},
    "108": {"label": "PVC-rich record", "db": "mitdb"},
    "201": {"label": "Atrial Fibrillation / PVC mix", "db": "mitdb"},
    "207": {"label": "Complete AV block", "db": "mitdb"},
    "217": {"label": "Bigeminy", "db": "mitdb"},
}

# Use only the first 30 seconds to keep demo fast (30 s × 360 Hz = 10 800 samples)
DEMO_DURATION_SAMPLES = 10_800


def list_samples() -> List[dict]:
    return [{"id": k, **v} for k, v in SAMPLE_RECORDS.items()]


def load_sample(record_id: str) -> dict:
    """
    Download (or load from cache) a PhysioNet record.

    Returns
    -------
    {
      "signal": list[float],   # raw MLII lead signal
      "sampling_rate": int,
      "record_id": str,
      "label": str,
    }
    """
    if record_id not in SAMPLE_RECORDS:
        raise ValueError(f"Unknown sample record: {record_id}")

    meta = SAMPLE_RECORDS[record_id]
    cache_path = SAMPLE_DIR / f"{record_id}.npy"
    sr_path = SAMPLE_DIR / f"{record_id}_sr.txt"

    if cache_path.exists() and sr_path.exists():
        signal = np.load(cache_path)
        sampling_rate = int(sr_path.read_text())
    else:
        record = wfdb.rdrecord(
            record_id,
            pn_dir=meta["db"],
            sampto=DEMO_DURATION_SAMPLES,
        )
        # Use MLII channel (channel 0 for all MIT-BIH records)
        signal = record.p_signal[:, 0].astype(np.float32)
        sampling_rate = record.fs
        np.save(cache_path, signal)
        sr_path.write_text(str(sampling_rate))

    return {
        "signal": signal.tolist(),
        "sampling_rate": int(sampling_rate),
        "record_id": record_id,
        "label": meta["label"],
    }
