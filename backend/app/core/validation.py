"""
Phase 6 — Validation utilities.

Computes and formats:
  - Per-class precision, recall, F1, support
  - Confusion matrix (label x label)
  - Patient-level accuracy (fraction of patients where dominant prediction is correct)
  - Serialises results to JSON-safe dicts for API responses and the training report
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


LABEL_ORDER = ["N", "AF", "PVC", "OTHER"]


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Returns a dict with keys:
      per_class: list of {label, precision, recall, f1, support}
      accuracy: float
      macro_f1: float
      weighted_f1: float
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = []
    for label in class_names:
        if label in report:
            r = report[label]
            per_class.append({
                "label": label,
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1-score"], 4),
                "support": int(r["support"]),
            })

    return {
        "per_class": per_class,
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }


def confusion_matrix_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Returns {labels: [...], matrix: [[...], ...]} — row=true, col=predicted.
    """
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": class_names,
        "matrix": cm.tolist(),
    }


def patient_split_report(
    patient_ids_test: List[str],
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Patient-level evaluation: each patient gets a dominant predicted label
    (majority vote over their beats). Reports per-patient accuracy.

    Parameters
    ----------
    patient_ids_test : record ID for each beat in the test set
    y_true_test      : ground-truth integer label per beat
    y_pred_test      : predicted integer label per beat
    class_names      : ordered list of class strings
    """
    patient_true: dict = defaultdict(list)
    patient_pred: dict = defaultdict(list)

    for pid, yt, yp in zip(patient_ids_test, y_true_test, y_pred_test):
        patient_true[pid].append(yt)
        patient_pred[pid].append(yp)

    patient_results = []
    correct = 0

    for pid in sorted(patient_true):
        # Dominant true label = most frequent in ground truth
        from collections import Counter
        true_dom = int(Counter(patient_true[pid]).most_common(1)[0][0])
        pred_dom = int(Counter(patient_pred[pid]).most_common(1)[0][0])
        is_correct = true_dom == pred_dom
        correct += int(is_correct)
        patient_results.append({
            "patient_id": pid,
            "n_beats": len(patient_true[pid]),
            "true_dominant": class_names[true_dom] if true_dom < len(class_names) else str(true_dom),
            "pred_dominant": class_names[pred_dom] if pred_dom < len(class_names) else str(pred_dom),
            "correct": is_correct,
        })

    patient_accuracy = correct / len(patient_results) if patient_results else 0.0

    return {
        "n_patients": len(patient_results),
        "patient_accuracy": round(patient_accuracy, 4),
        "patients": patient_results,
    }


def print_validation_report(
    split_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    patient_ids: List[str] | None = None,
) -> None:
    """Print a formatted validation report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {split_label} Evaluation  ({len(y_true)} beats)")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "         " + "  ".join(f"{c:>6}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {class_names[i]:>6}  {row_str}")

    if patient_ids is not None:
        pat = patient_split_report(patient_ids, y_true, y_pred, class_names)
        print(f"\nPatient-level accuracy: {pat['patient_accuracy']:.4f} "
              f"({sum(p['correct'] for p in pat['patients'])}/{pat['n_patients']} patients correct)")
