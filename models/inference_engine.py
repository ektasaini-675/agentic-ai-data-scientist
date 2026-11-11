"""
Real-Time Inference Engine & Counterfactual Explainer.
Loads serialized model artifacts from `models/best_model.joblib` to execute
sub-millisecond predictions, class probability distributions, and what-if sensitivity analysis.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class InferenceEngine:
    def __init__(self, artifact_path: str = "models/best_model.joblib"):
        self.artifact_path = artifact_path
        self.artifact = None
        self.model = None
        self.feature_names = []
        self.classes_map = {}
        self.problem_type = "classification"
        self.scaler = None
        self.load()

    def load(self) -> bool:
        if not os.path.exists(self.artifact_path):
            return False
        try:
            self.artifact = joblib.load(self.artifact_path)
            self.model = self.artifact.get("model")
            self.feature_names = self.artifact.get("feature_names", [])
            self.classes_map = self.artifact.get("classes_map", {})
            self.problem_type = self.artifact.get("problem_type", "classification")
            self.scaler = self.artifact.get("scaler")
            return True
        except Exception:
            return False

    def is_ready(self) -> bool:
        return self.model is not None and len(self.feature_names) > 0

    def predict_one(self, feature_values: Dict[str, Any]) -> Dict[str, Any]:
        """Runs single-instance inference with feature alignment and probabilities."""
        if not self.is_ready():
            return {"error": "Inference engine model artifact not loaded."}

        # Build ordered vector
        vector = []
        for feat in self.feature_names:
            val = feature_values.get(feat, 0.0)
            try:
                vector.append(float(val))
            except (ValueError, TypeError):
                vector.append(0.0)

        X = np.array([vector], dtype=np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.problem_type == "classification":
            pred_idx = int(self.model.predict(X)[0])
            pred_label = self.classes_map.get(pred_idx, str(pred_idx))
            
            probs_dict = {}
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                for idx, p in enumerate(probs):
                    label = self.classes_map.get(idx, str(idx))
                    probs_dict[label] = round(float(p), 4)

            return {
                "prediction": pred_label,
                "prediction_index": pred_idx,
                "probabilities": probs_dict,
                "confidence": round(float(max(probs_dict.values())), 4) if probs_dict else 1.0,
                "model_name": self.artifact.get("model_name")
            }
        else:
            pred_val = float(self.model.predict(X)[0])
            return {
                "prediction": round(pred_val, 4),
                "model_name": self.artifact.get("model_name"),
                "problem_type": "regression"
            }

    def counterfactual_sensitivity(
        self,
        base_features: Dict[str, Any],
        top_k: int = 5,
        perturbation_pct: float = 0.20
    ) -> List[Dict[str, Any]]:
        """
        Evaluates what happens to the output probability if each top feature is perturbed by +/- 20%.
        Returns actionable counterfactual sensitivity insights.
        """
        if not self.is_ready() or self.problem_type != "classification":
            return []

        base_res = self.predict_one(base_features)
        base_pred = base_res.get("prediction")
        base_probs = base_res.get("probabilities", {})

        top_feats = [f["feature"] for f in self.artifact.get("top_features", [])[:top_k]]
        if not top_feats:
            top_feats = self.feature_names[:top_k]

        sensitivities = []
        for feat in top_feats:
            val = float(base_features.get(feat, 0.0))
            if val == 0:
                delta = 1.0
            else:
                delta = abs(val) * perturbation_pct

            # Test increase
            feat_up = base_features.copy()
            feat_up[feat] = val + delta
            res_up = self.predict_one(feat_up)

            # Test decrease
            feat_down = base_features.copy()
            feat_down[feat] = val - delta
            res_down = self.predict_one(feat_down)

            # Sensitivity impact
            prob_shift_up = res_up.get("probabilities", {}).get(base_pred, 0.5) - base_probs.get(base_pred, 0.5)
            prob_shift_down = res_down.get("probabilities", {}).get(base_pred, 0.5) - base_probs.get(base_pred, 0.5)

            sensitivities.append({
                "feature": feat,
                "baseline_value": round(val, 3),
                "delta_tested": round(delta, 3),
                "prob_change_on_increase": round(prob_shift_up, 4),
                "prob_change_on_decrease": round(prob_shift_down, 4),
                "prediction_flips_on_increase": res_up.get("prediction") != base_pred,
                "prediction_flips_on_decrease": res_down.get("prediction") != base_pred,
                "sensitivity_magnitude": round(abs(prob_shift_up) + abs(prob_shift_down), 4)
            })

        sensitivities.sort(key=lambda x: -x["sensitivity_magnitude"])
        return sensitivities

