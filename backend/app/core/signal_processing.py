"""
Signal pre-processing pipeline.

Steps:
  1. Bandpass filter  (0.5-40 Hz)  -- removes baseline wander and HF noise
  2. R-peak detection via neurokit2  -- Pan-Tompkins algorithm
  3. Beat segmentation  -- asymmetric window: PRE_SAMPLES before / POST_SAMPLES after R-peak
     Methodology-compliant: 100 samples before, 200 after (at 360 Hz = ~278ms / ~556ms)
  4. R-peak validation against annotations (training path only)
"""
from __future__ import annotations

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, sosfiltfilt
from typing import List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Window constants (methodology: 100 pre / 200 post at any sampling rate)
# These are scaled proportionally when fs != 360
# --------------------------------------------------------------------------- #
PRE_SAMPLES_360 = 100    # at 360 Hz
POST_SAMPLES_360 = 200   # at 360 Hz
LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0


def bandpass_filter(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """4th-order Butterworth bandpass filter, 0.5-40 Hz."""
    nyq = sampling_rate / 2.0
    low = LOWCUT_HZ / nyq
    high = min(HIGHCUT_HZ / nyq, 0.99)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def detect_rpeaks(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Pan-Tompkins R-peak detection via neurokit2.
    Returns array of R-peak sample indices.
    """
    _, info = nk.ecg_peaks(signal, sampling_rate=sampling_rate, method="pantompkins1985")
    return info["ECG_R_Peaks"]


def validate_rpeaks(
    detected: np.ndarray,
    annotated: np.ndarray,
    tolerance_samples: int = 50,
) -> dict:
    """
    Compare detected R-peaks against cardiologist annotations.

    Parameters
    ----------
    detected   : indices from Pan-Tompkins
    annotated  : ground-truth R-peak indices from WFDB annotation file
    tolerance  : window (samples) within which a detection counts as TP

    Returns
    -------
    dict with sensitivity, ppv (precision), f1, n_tp, n_fp, n_fn
    """
    tp, fp, fn = 0, 0, 0
    matched = set()

    for det in detected:
        diffs = np.abs(annotated - det)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tolerance_samples and idx not in matched:
            tp += 1
            matched.add(idx)
        else:
            fp += 1

    fn = len(annotated) - len(matched)
    sensitivity = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    f1 = 2 * sensitivity * ppv / max(sensitivity + ppv, 1e-9)

    return {"sensitivity": sensitivity, "ppv": ppv, "f1": f1,
            "n_tp": tp, "n_fp": fp, "n_fn": fn}


def segment_beats(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    sampling_rate: int,
) -> Tuple[List[np.ndarray], List[int], List[int], List[int]]:
    """
    Extract asymmetric fixed-length windows around each R-peak.
    Window = PRE_SAMPLES_360 * (fs/360) before R, POST_SAMPLES_360 * (fs/360) after R.

    Returns
    -------
    beats       : list of 1-D arrays
    valid_peaks : R-peak sample indices (boundary-filtered)
    starts      : window start sample for each beat (for frontend overlay)
    ends        : window end sample for each beat
    """
    scale = sampling_rate / 360.0
    pre = int(PRE_SAMPLES_360 * scale)
    post = int(POST_SAMPLES_360 * scale)
    n = len(signal)

    beats, valid_peaks, starts, ends = [], [], [], []

    for idx in rpeaks:
        start = int(idx) - pre
        end = int(idx) + post
        if start < 0 or end > n:
            continue
        beats.append(signal[start:end].copy())
        valid_peaks.append(int(idx))
        starts.append(start)
        ends.append(end)

    return beats, valid_peaks, starts, ends


def process_signal(
    raw_signal: np.ndarray,
    sampling_rate: int,
    ann_rpeaks: Optional[np.ndarray] = None,
) -> dict:
    """
    Full pre-processing pipeline.

    Parameters
    ----------
    raw_signal    : raw ECG signal (mV)
    sampling_rate : Hz
    ann_rpeaks    : ground-truth R-peak indices (optional; enables validation)

    Returns
    -------
    dict with keys:
      filtered, rpeaks, beats, beat_starts, beat_ends, sampling_rate,
      rpeak_validation (if ann_rpeaks provided)
    """
    filtered = bandpass_filter(raw_signal, sampling_rate)
    rpeaks_raw = detect_rpeaks(filtered, sampling_rate)
    beats, rpeaks, starts, ends = segment_beats(filtered, rpeaks_raw, sampling_rate)

    result = {
        "filtered": filtered,
        "rpeaks": rpeaks,
        "beats": beats,
        "beat_starts": starts,
        "beat_ends": ends,
        "sampling_rate": sampling_rate,
    }

    if ann_rpeaks is not None and len(ann_rpeaks) > 0:
        result["rpeak_validation"] = validate_rpeaks(rpeaks_raw, ann_rpeaks)

    return result
