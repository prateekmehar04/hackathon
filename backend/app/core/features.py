"""
Feature extraction from segmented ECG beats.

Features per beat (32 total):
  RR / HRV (7)     -- curr_rr, prev_rr, rr_ratio, rr_variability, heart_rate,
                       rmssd, pnn50
  Morphology (17)  -- R/Q/S/P/T amplitudes, QRS duration, PR interval, QT interval,
                       R-rise slope, R-fall slope, ST elevation, T symmetry,
                       P duration, T duration, beat area, beat norm area, p_present
  Statistics (8)   -- mean, std, skewness, kurtosis, energy, zero crossings,
                       max abs, waveform entropy

The beat window is asymmetric: PRE samples before R-peak, POST samples after.
R-peak is located at index PRE (= 100 * fs/360).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from typing import List, Optional


FEATURE_NAMES = [
    # RR / HRV
    "curr_rr_ms",
    "prev_rr_ms",
    "rr_ratio",
    "rr_variability_ms",
    "heart_rate_bpm",
    "rmssd_ms",
    "pnn50",
    # Morphology
    "r_amplitude",
    "q_amplitude",
    "s_amplitude",
    "p_amplitude",
    "t_amplitude",
    "p_present",
    "qrs_duration_ms",
    "pr_interval_ms",
    "qt_interval_ms",
    "r_rise_slope",
    "r_fall_slope",
    "st_elevation",
    "t_symmetry",
    "p_duration_ms",
    "t_duration_ms",
    "beat_area",
    "beat_norm_area",
    # Statistics
    "beat_mean",
    "beat_std",
    "beat_skew",
    "beat_kurt",
    "beat_energy",
    "zero_crossings",
    "max_abs",
    "waveform_entropy",
]

N_FEATURES = len(FEATURE_NAMES)  # 32


def _safe(val: float, fallback: float = 0.0) -> float:
    """Return val if finite, else fallback."""
    return float(val) if np.isfinite(val) else fallback


def _rr_features(rpeaks: List[int], sampling_rate: int, beat_index: int) -> List[float]:
    """
    RR-interval features for beat at beat_index.

    Features
    --------
    curr_rr_ms      : RR interval ending at this beat
    prev_rr_ms      : RR interval ending at previous beat
    rr_ratio        : curr_rr / prev_rr  (>1 = longer than previous, AFib signal)
    rr_variability  : |curr_rr - prev_rr|  (local beat-to-beat variability)
    heart_rate_bpm  : 60000 / curr_rr
    rmssd_ms        : root-mean-square successive differences over full recording
    pnn50           : fraction of successive RR diffs > 50 ms
    """
    n = len(rpeaks)
    if n < 2:
        return [0.0, 0.0, 1.0, 0.0, 75.0, 0.0, 0.0]

    rr_samples = np.diff(rpeaks)
    rr_ms = rr_samples * 1000.0 / sampling_rate

    i = beat_index
    # curr_rr: interval just before beat i (index i-1 in diff array)
    if i > 0 and i <= len(rr_ms):
        curr_rr = _safe(rr_ms[i - 1])
    else:
        curr_rr = _safe(float(np.mean(rr_ms)))

    # prev_rr: interval before that (index i-2)
    if i > 1 and (i - 2) < len(rr_ms):
        prev_rr = _safe(rr_ms[i - 2])
    else:
        prev_rr = curr_rr  # no previous — treat as same

    rr_ratio = _safe(curr_rr / prev_rr) if prev_rr > 0 else 1.0
    rr_variability = _safe(abs(curr_rr - prev_rr))
    hr = _safe(60_000.0 / curr_rr) if curr_rr > 0 else 75.0

    # Recording-wide RMSSD
    rmssd = _safe(float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))) if len(rr_ms) > 1 else 0.0

    # pNN50
    nn50 = int(np.sum(np.abs(np.diff(rr_ms)) > 50))
    pnn50 = _safe(nn50 / len(rr_ms)) if len(rr_ms) > 1 else 0.0

    return [curr_rr, prev_rr, rr_ratio, rr_variability, hr, rmssd, pnn50]


def _morphology_features(beat: np.ndarray, sampling_rate: int, pre_samples: int) -> List[float]:
    """
    Morphological features from the beat window.

    Parameters
    ----------
    beat        : 1-D signal window (length = pre + post)
    sampling_rate
    pre_samples : number of samples before R-peak (R-peak is at beat[pre_samples])
    """
    n = len(beat)
    r_idx = pre_samples  # R-peak position within the window

    r_amp = _safe(beat[r_idx])

    # --- Q wave: minimum in 30 ms window before R ---
    q_win = max(1, int(0.030 * sampling_rate))
    q_region = beat[max(0, r_idx - q_win): r_idx]
    q_amp = _safe(float(np.min(q_region))) if len(q_region) > 0 else 0.0

    # --- S wave: minimum in 40 ms window after R ---
    s_win = max(1, int(0.040 * sampling_rate))
    s_region = beat[r_idx: min(n, r_idx + s_win)]
    s_amp = _safe(float(np.min(s_region))) if len(s_region) > 0 else 0.0

    # --- P wave: peak in 50–200 ms before R ---
    p_end = max(0, r_idx - int(0.050 * sampling_rate))
    p_start = max(0, r_idx - int(0.200 * sampling_rate))
    p_region = beat[p_start:p_end]
    if len(p_region) > 2:
        p_peaks, _ = find_peaks(p_region, height=0)
        if len(p_peaks):
            p_amp = _safe(float(p_region[p_peaks[np.argmax(p_region[p_peaks])]]))
            p_present = 1.0
        else:
            # Small positive deflection counts
            pmax = float(np.max(p_region))
            p_amp = _safe(pmax)
            p_present = 1.0 if pmax > 0.05 * abs(r_amp) else 0.0
    else:
        p_amp = 0.0
        p_present = 0.0

    # --- T wave: peak in 100–400 ms after R ---
    t_start = r_idx + int(0.100 * sampling_rate)
    t_end = min(n, r_idx + int(0.400 * sampling_rate))
    t_region = beat[t_start:t_end]
    if len(t_region) > 2:
        t_peaks, _ = find_peaks(t_region)
        t_amp = _safe(float(t_region[t_peaks[0]])) if len(t_peaks) else _safe(float(np.max(t_region)))
    else:
        t_amp = 0.0

    # --- QRS duration: width where |amplitude| > 50% R amplitude ---
    qrs_half = int(0.060 * sampling_rate)
    qrs_region = beat[max(0, r_idx - qrs_half): min(n, r_idx + qrs_half)]
    thresh = 0.5 * abs(r_amp)
    qrs_dur_ms = _safe(float(np.sum(np.abs(qrs_region) > thresh)) * 1000.0 / sampling_rate)

    # --- PR interval: P-peak to R ---
    if p_present and len(p_region) > 2:
        p_peaks2, _ = find_peaks(p_region, height=0)
        if len(p_peaks2):
            p_peak_abs = p_start + p_peaks2[np.argmax(p_region[p_peaks2])]
            pr_ms = _safe((r_idx - p_peak_abs) * 1000.0 / sampling_rate)
        else:
            pr_ms = _safe((r_idx - p_end) * 1000.0 / sampling_rate)
    else:
        pr_ms = 0.0

    # --- QT interval: R to end of T ---
    if len(t_region) > 2:
        t_end_idx = t_start + min(len(t_region) - 1, int(0.350 * sampling_rate))
        qt_ms = _safe((t_end_idx - r_idx) * 1000.0 / sampling_rate)
    else:
        qt_ms = 0.0

    # --- R slopes (mV/sample) ---
    slope_win = max(2, int(0.010 * sampling_rate))  # 10 ms window
    r_rise = _safe((beat[r_idx] - beat[max(0, r_idx - slope_win)]) / slope_win)
    r_fall = _safe((beat[r_idx] - beat[min(n - 1, r_idx + slope_win)]) / slope_win)

    # --- ST elevation: mean 60-80 ms after R ---
    st_start = r_idx + int(0.060 * sampling_rate)
    st_end = r_idx + int(0.080 * sampling_rate)
    st_seg = beat[min(n, st_start): min(n, st_end)]
    st_elev = _safe(float(np.mean(st_seg))) if len(st_seg) > 0 else 0.0

    # --- T-wave symmetry: area left/right of T peak ---
    if len(t_region) > 4 and len(find_peaks(t_region)[0]) > 0:
        t_pk = t_start + find_peaks(t_region)[0][0]
        t_left_area = float(np.sum(np.abs(beat[t_start: t_pk + 1])))
        t_right_area = float(np.sum(np.abs(beat[t_pk: min(n, t_pk + (t_pk - t_start) + 1)])))
        t_sym = _safe(t_left_area / (t_right_area + 1e-9))
    else:
        t_sym = 1.0

    # --- P wave duration ---
    p_dur_ms = _safe(len(p_region) * 1000.0 / sampling_rate)

    # --- T wave duration ---
    t_dur_ms = _safe(len(t_region) * 1000.0 / sampling_rate)

    # --- Beat area and normalised area ---
    beat_area = _safe(float(np.trapezoid(beat)))
    beat_norm_area = _safe(beat_area / (n + 1e-9))

    return [
        r_amp, q_amp, s_amp, p_amp, t_amp, p_present,
        qrs_dur_ms, pr_ms, qt_ms,
        r_rise, r_fall, st_elev, t_sym,
        p_dur_ms, t_dur_ms, beat_area, beat_norm_area,
    ]


def _statistical_features(beat: np.ndarray) -> List[float]:
    """Statistical descriptors of the beat window."""
    energy = _safe(float(np.sum(beat ** 2)))
    zc = int(np.sum(np.diff(np.sign(beat)) != 0))
    prob, _ = np.histogram(beat, bins=10, density=True)
    prob = prob[prob > 0]
    entropy = _safe(float(-np.sum(prob * np.log(prob + 1e-12))))

    return [
        _safe(float(np.mean(beat))),
        _safe(float(np.std(beat))),
        _safe(float(skew(beat))),
        _safe(float(kurtosis(beat))),
        energy,
        float(zc),
        _safe(float(np.max(np.abs(beat)))),
        entropy,
    ]


def extract_features(
    beats: List[np.ndarray],
    rpeaks: List[int],
    sampling_rate: int,
    pre_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Extract a feature matrix of shape (n_beats, N_FEATURES).

    Parameters
    ----------
    beats         : beat windows from segment_beats()
    rpeaks        : R-peak sample indices (same length as beats)
    sampling_rate : Hz
    pre_samples   : samples before R-peak in each window; inferred from fs if None
    """
    if pre_samples is None:
        pre_samples = int(100 * sampling_rate / 360)

    rows = []
    for i, beat in enumerate(beats):
        hrv = _rr_features(rpeaks, sampling_rate, i)
        morph = _morphology_features(beat, sampling_rate, pre_samples)
        stat = _statistical_features(beat)
        rows.append(hrv + morph + stat)

    if not rows:
        return np.empty((0, N_FEATURES), dtype=np.float32)

    arr = np.array(rows, dtype=np.float32)
    assert arr.shape[1] == N_FEATURES, f"Feature count mismatch: {arr.shape[1]} vs {N_FEATURES}"
    return arr


