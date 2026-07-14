"""
Offline training script  --  follows the full methodology:

  Phase 1: MIT-BIH data acquisition
  Phase 2: Signal pre-processing (bandpass, Pan-Tompkins, asymmetric beat window)
  Phase 3: Feature engineering (RR current/prev/ratio/variability, morphology, stats)
  Phase 4: PATIENT-LEVEL train/test split (no data leakage), class-balanced RF + XGB
  Phase 5: SHAP explainability baked in at inference time
  Phase 6: Per-class metrics, confusion matrix, patient-level accuracy report

Usage
-----
    cd backend
    python train.py                         # all 44 records, RF + XGB
    python train.py --records 100 108 201   # subset
    python train.py --skip-xgb             # RF only (faster)
    python train.py --test-frac 0.25       # 25 percent of patients held out

Label mapping (AAMI EC57 simplified, 4-class):
    N     <-- N, L, R, e, j          Normal / bundle-branch aberrations
    PVC   <-- V, E                   Premature Ventricular Contraction
    AF    <-- A, a, J, S             Supraventricular ectopic (AF / PAC proxy)
    OTHER <-- F, f, /, Q, ?          Fusion, paced, unclassifiable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import wfdb
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))

from app.core.signal_processing import process_signal
from app.core.features import extract_features, N_FEATURES
from app.core.validation import (
    per_class_report, confusion_matrix_report,
    patient_split_report, print_validation_report,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_DIR = Path(__file__).parent / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# All 44 usable MIT-BIH records (records 102, 104, 107, 217 excluded: paced)
DEFAULT_RECORDS = [
    "100", "101", "103", "105", "106", "108",
    "109", "111", "112", "113", "114", "115",
    "116", "117", "118", "119", "121", "122",
    "123", "124", "200", "201", "202", "203",
    "205", "207", "208", "209", "210", "212",
    "213", "214", "215", "219", "220", "221",
    "222", "223", "228", "230", "231", "232",
    "233", "234",
]

# AAMI EC57 label map (4-class)
LABEL_MAP = {
    "N": "N",  "L": "N",  "R": "N",  "e": "N",  "j": "N",
    "V": "PVC","E": "PVC",
    "A": "AF", "a": "AF", "J": "AF", "S": "AF",
    "F": "OTHER", "f": "OTHER", "/": "OTHER", "Q": "OTHER", "?": "OTHER",
}
CLASSES = ["N", "AF", "PVC", "OTHER"]  # intentional order for display

DEMO_DURATION = 650_000   # ~30 min at 360 Hz


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_record(rid: str):
    """Download (or cache) one MIT-BIH record. Returns (signal, ann_samples, ann_symbols, fs)."""
    print(f"  Loading {rid}...", end=" ", flush=True)
    rec = wfdb.rdrecord(rid, pn_dir="mitdb", sampto=DEMO_DURATION)
    ann = wfdb.rdann(rid, "atr", pn_dir="mitdb", sampto=DEMO_DURATION)
    signal = rec.p_signal[:, 0].astype(np.float32)
    print(f"{len(ann.sample)} annotations")
    return signal, ann.sample, ann.symbol, int(rec.fs)


def build_dataset(records: list[str], validate_peaks: bool = True):
    """
    Process all records, extract features, map labels.

    Returns
    -------
    X          : (n_beats, N_FEATURES) float32
    y_str      : (n_beats,) string labels
    beat_pids  : (n_beats,) record IDs (for patient-level split)
    val_stats  : dict of R-peak validation results per record
    """
    all_X, all_y, all_pids = [], [], []
    val_stats = {}

    for rid in records:
        try:
            signal, ann_samples, ann_symbols, fs = load_record(rid)
        except Exception as exc:
            print(f"  WARN: skipping {rid} -- {exc}")
            continue

        # Ground-truth R-peak positions (beat annotations only, not rhythm changes)
        beat_ann_mask = np.array([s in LABEL_MAP for s in ann_symbols])
        ann_rpeaks = ann_samples[beat_ann_mask]
        ann_syms = np.array(ann_symbols)[beat_ann_mask]

        # Pre-process signal; pass annotation R-peaks for validation
        proc = process_signal(signal, fs, ann_rpeaks=ann_rpeaks)
        beats = proc["beats"]
        rpeaks = proc["rpeaks"]

        if validate_peaks and "rpeak_validation" in proc:
            v = proc["rpeak_validation"]
            val_stats[rid] = v
            # Skip records where detection is unreliable
            if v["sensitivity"] < 0.85:
                print(f"    WARN: {rid} sensitivity={v['sensitivity']:.3f} -- skipping")
                continue

        if len(beats) == 0:
            continue

        pre_samples = int(100 * fs / 360)
        X = extract_features(beats, rpeaks, fs, pre_samples=pre_samples)

        # Assign label to each detected beat by proximity to annotation
        ann_map = dict(zip(ann_samples, ann_symbols))
        beat_labels = []
        for rpeak in rpeaks:
            diffs = np.abs(ann_samples - rpeak)
            closest_idx = int(np.argmin(diffs))
            if diffs[closest_idx] <= 50:
                sym = ann_symbols[closest_idx]
                label = LABEL_MAP.get(sym, "OTHER")
            else:
                label = "N"   # unlabelled beat near no annotation -- treat as normal
            beat_labels.append(label)

        all_X.append(X)
        all_y.extend(beat_labels)
        all_pids.extend([rid] * len(beat_labels))

    X_all = np.vstack(all_X)
    y_str = np.array(all_y)
    beat_pids = np.array(all_pids)

    print(f"\nDataset: {X_all.shape[0]} beats from {len(set(all_pids))} patients")
    class_dist = {lbl: int(c) for lbl, c in zip(*np.unique(y_str, return_counts=True))}
    print(f"Class distribution: {class_dist}")

    return X_all, y_str, beat_pids, val_stats


# --------------------------------------------------------------------------- #
# Patient-level train/test split
# --------------------------------------------------------------------------- #
def patient_split(
    beat_pids: np.ndarray,
    y_str: np.ndarray,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split beats by patient (record ID), not by individual beat.

    This prevents data leakage: beats from the same patient are correlated
    (same QRS morphology, same noise level, etc.). Testing on unseen patients
    gives an honest accuracy estimate.

    Returns train_mask, test_mask (boolean arrays over beats).
    """
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(set(beat_pids)))
    rng.shuffle(patients)

    n_test = max(1, int(len(patients) * test_frac))
    test_patients = set(patients[:n_test])
    train_patients = set(patients[n_test:])

    train_mask = np.array([p in train_patients for p in beat_pids])
    test_mask = np.array([p in test_patients for p in beat_pids])

    print(f"\nPatient split: {len(train_patients)} train / {len(test_patients)} test patients")
    print(f"  Train patients: {sorted(train_patients)}")
    print(f"  Test  patients: {sorted(test_patients)}")

    return train_mask, test_mask


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(records: list[str], test_frac: float = 0.20, skip_xgb: bool = False):
    print("\n=== Phase 1+2+3: Build dataset ===")
    X, y_str, beat_pids, val_stats = build_dataset(records)

    # Print R-peak validation summary
    if val_stats:
        print("\nR-peak validation summary:")
        sensitivities = [v["sensitivity"] for v in val_stats.values()]
        ppvs = [v["ppv"] for v in val_stats.values()]
        print(f"  Mean sensitivity: {np.mean(sensitivities):.4f}  "
              f"Mean PPV: {np.mean(ppvs):.4f}  "
              f"Min sensitivity: {np.min(sensitivities):.4f}")

    # Encode labels
    label_enc = LabelEncoder()
    label_enc.fit(CLASSES)
    # Map any label not in CLASSES to OTHER
    y_str_clamped = np.array([s if s in CLASSES else "OTHER" for s in y_str])
    y = label_enc.transform(y_str_clamped)
    joblib.dump(label_enc, MODEL_DIR / "label_encoder.joblib")

    print("\n=== Phase 4: Patient-level train/test split ===")
    train_mask, test_mask = patient_split(beat_pids, y_str, test_frac=test_frac)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    pids_test = beat_pids[test_mask]

    print(f"  Train beats: {len(y_train)}  Test beats: {len(y_test)}")

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # Class weights to handle imbalance (Normal >> PVC >> AF >> OTHER)
    classes_present = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes_present, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, cw)}
    print(f"\n  Class weights (balanced): {class_weight_dict}")

    print("\n=== Training Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, MODEL_DIR / "rf_model.joblib")
    print("  Saved -> app/models/rf_model.joblib")

    # Phase 6: Validation
    print("\n=== Phase 6: Validation ===")
    y_pred_train = rf.predict(X_train_s)
    y_pred_test = rf.predict(X_test_s)

    print_validation_report("TRAIN", y_train, y_pred_train, label_enc.classes_.tolist())
    print_validation_report("TEST (patient-held-out)", y_test, y_pred_test,
                            label_enc.classes_.tolist(), patient_ids=pids_test.tolist())

    # Save validation results as JSON for the API
    test_report = per_class_report(y_test, y_pred_test, label_enc.classes_.tolist())
    cm_report = confusion_matrix_report(y_test, y_pred_test, label_enc.classes_.tolist())
    pat_report = patient_split_report(
        pids_test.tolist(), y_test.tolist(), y_pred_test.tolist(),
        label_enc.classes_.tolist()
    )

    validation_results = {
        "model": "rf",
        "n_train_beats": int(len(y_train)),
        "n_test_beats": int(len(y_test)),
        "n_train_patients": int(np.sum(train_mask > 0) > 0),  # count unique
        "train_patients": sorted(set(beat_pids[train_mask].tolist())),
        "test_patients": sorted(set(pids_test.tolist())),
        "rpeak_validation": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                  for kk, vv in v.items()}
                              for k, v in val_stats.items()},
        "test_metrics": test_report,
        "confusion_matrix": cm_report,
        "patient_report": pat_report,
    }
    val_path = MODEL_DIR / "validation_results.json"
    val_path.write_text(json.dumps(validation_results, indent=2))
    print(f"\nValidation results saved -> {val_path}")

    if not skip_xgb:
        print("\n=== Training XGBoost ===")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            n_jobs=-1,
            random_state=42,
            device="cpu",
        )
        xgb_model.fit(
            X_train_s, y_train,
            sample_weight=np.array([class_weight_dict.get(int(yi), 1.0) for yi in y_train]),
        )
        y_pred_xgb = xgb_model.predict(X_test_s)
        print_validation_report("TEST XGB (patient-held-out)", y_test, y_pred_xgb,
                                 label_enc.classes_.tolist(), patient_ids=pids_test.tolist())
        joblib.dump(xgb_model, MODEL_DIR / "xgb_model.joblib")
        print("  Saved -> app/models/xgb_model.joblib")

    print("\nTraining complete. Artifacts in app/models/")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia classifier")
    parser.add_argument("--records", nargs="+", default=DEFAULT_RECORDS)
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction of patients held out for test (default 0.20)")
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()
    train(args.records, test_frac=args.test_frac, skip_xgb=args.skip_xgb)
