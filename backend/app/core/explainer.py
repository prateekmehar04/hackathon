"""
SHAP-based explainability module.

For each classified beat, computes:
  - shap_values  – per-feature contribution to the predicted class
  - top_features – top-N (feature_name, shap_value, direction) triples

Uses TreeExplainer for Random Forest / XGBoost (fast, exact).
"""
from __future__ import annotations

import numpy as np
import shap
from typing import List

from app.core.features import FEATURE_NAMES


class SHAPExplainer:
    """Wraps shap.TreeExplainer for the trained classifier."""

    def __init__(self, classifier):
        # tree_path_dependent is model-only (no background data needed)
        self.explainer = shap.TreeExplainer(
            classifier.model,
            feature_perturbation="tree_path_dependent",
        )
        self.scaler = classifier.scaler
        self.label_encoder = classifier.label_encoder

    def explain_batch(
        self,
        X: np.ndarray,
        predicted_class_indices: List[int],
        top_n: int = 5,
    ) -> List[dict]:
        """
        Return SHAP explanation for each beat.

        Parameters
        ----------
        X                      : raw (unscaled) feature matrix (n_beats, n_features)
        predicted_class_indices: integer class index for each beat
        top_n                  : number of top features to return

        Returns
        -------
        List of dicts, one per beat:
          {
            "shap_values": [...],          # float list, one per feature
            "base_value":  float,
            "top_features": [
              {"name": str, "value": float, "shap": float, "direction": "positive"|"negative"},
              ...
            ]
          }
        """
        Xs = self.scaler.transform(X)

        # shap_values returns an Explanation object in shap >= 0.44
        explanation = self.explainer(Xs)

        # .values shape: (n_samples, n_features) for binary
        #                (n_samples, n_features, n_classes) for multi-class
        sv_all = explanation.values
        base_all = explanation.base_values

        results = []
        for i, class_idx in enumerate(predicted_class_indices):
            if sv_all.ndim == 3:
                sv = sv_all[i, :, class_idx]
                base = float(base_all[i, class_idx]) if base_all.ndim == 2 else float(base_all[i])
            else:
                sv = sv_all[i]
                base = float(base_all[i]) if np.ndim(base_all) > 0 else float(base_all)

            # Top-N by absolute magnitude
            order = np.argsort(np.abs(sv))[::-1][:top_n]
            top = [
                {
                    "name": FEATURE_NAMES[j],
                    "value": float(X[i, j]),
                    "shap": float(sv[j]),
                    "direction": "positive" if sv[j] >= 0 else "negative",
                }
                for j in order
            ]

            results.append(
                {
                    "shap_values": sv.tolist(),
                    "base_value": base,
                    "top_features": top,
                }
            )

        return results
