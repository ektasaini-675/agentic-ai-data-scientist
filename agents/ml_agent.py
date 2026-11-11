"""
Machine Learning & Predictive Modeling Agent.
Performs automated multi-model benchmarking, 5-Fold Stratified Cross-Validation,
class imbalance mitigation, Explainable AI (XAI) feature attribution with directional impact,
and adaptive regularization remediation in response to Critic Agent directives.
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from agents.state import AgentState

COMMON_TARGET_NAMES = {
    "target", "class", "label", "outcome", "status", "result",
    "churn", "y", "diagnosis", "default", "approved", "loan_status",
    "heartdisease", "heart_disease", "condition"
}


def _guess_target_column(df: pd.DataFrame, goal: str, id_cols: List[str]) -> str:
    goal_words = set(re.findall(r"[a-z0-9_]+", goal.lower()))

    candidates = []
    for col in df.columns:
        if col in id_cols:
            continue
        col_words = set(re.findall(r"[a-z0-9]+", col.lower()))
        overlap = len(col_words & goal_words)
        if overlap > 0:
            candidates.append((overlap, col))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    for col in df.columns:
        if col.lower() in COMMON_TARGET_NAMES and col not in id_cols:
            return col

    non_id_cols = [c for c in df.columns if c not in id_cols]
    return non_id_cols[-1] if non_id_cols else df.columns[-1]


def run(state: AgentState) -> AgentState:
    data_path = state.get("cleaned_dataset_path") or state["dataset_path"]
    df = pd.read_csv(data_path)

    id_cols = state.get("schema_report", {}).get("id_columns", [])
    target_col = _guess_target_column(df, state.get("user_goal", ""), id_cols)

    # 1. Parse Critic Directives for Adaptive Remediation
    directives = state.get("critic_directives", [])
    regularize_flag = False
    custom_max_depth = 6
    custom_min_samples = 2
    l2_c = 1.0

    for d in directives:
        if d.get("action") == "regularize_model":
            regularize_flag = True
            custom_max_depth = d.get("max_depth", 3)
            custom_min_samples = d.get("min_samples_split", 10)
            l2_c = d.get("l2_penalty", 0.1)
            state.setdefault("agent_trace", []).append(
                f"[ml_agent] ADAPTIVE REMEDIATION APPLIED: Regularizing models (max_depth={custom_max_depth}, min_samples={custom_min_samples}, C={l2_c}) to suppress overfitting."
            )

    feature_cols = [c for c in df.columns if c != target_col and c not in id_cols]
    X_df = df[feature_cols].copy()
    y_series = df[target_col].copy()

    # One-hot encode categorical features
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    if X_df.shape[1] == 0:
        state["model_report"] = {"status": "skipped", "reason": "No predictive features available."}
        return state

    for col in X_df.columns:
        if X_df[col].dtype == bool:
            X_df[col] = X_df[col].astype(int)

    feature_names = list(X_df.columns)
    X = np.asarray(X_df.values, dtype=np.float32)

    is_classification = y_series.nunique() <= 15 or y_series.dtype == object

    if is_classification:
        # Encode target
        if not pd.api.types.is_numeric_dtype(y_series):
            le = LabelEncoder()
            y = np.asarray(le.fit_transform(y_series.astype(str)), dtype=np.int32)
            classes_map = {idx: str(cls) for idx, cls in enumerate(le.classes_)}
        else:
            y = np.asarray(y_series.values, dtype=np.int32)
            classes_map = {int(v): str(v) for v in np.unique(y)}

        # Class Imbalance Audit
        class_counts = pd.Series(y).value_counts()
        majority_count = class_counts.max()
        minority_count = class_counts.min()
        imbalance_ratio = round(majority_count / max(1, minority_count), 2)
        state["class_imbalance_ratio"] = imbalance_ratio

        class_weight_setting = "balanced" if imbalance_ratio > 1.8 else None

        # Train/Test Split (Stratified 80/20)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Standard Scaling for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Candidate Model Portfolio
        models = {
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=100, max_depth=custom_max_depth, min_samples_split=custom_min_samples,
                class_weight=class_weight_setting, random_state=42
            ),
            "GradientBoostingClassifier": GradientBoostingClassifier(
                n_estimators=80, max_depth=min(custom_max_depth, 4), random_state=42
            ),
            "ExtraTreesClassifier": ExtraTreesClassifier(
                n_estimators=100, max_depth=custom_max_depth, min_samples_split=custom_min_samples,
                class_weight=class_weight_setting, random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000, C=l2_c, class_weight=class_weight_setting, random_state=42
            )
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        leaderboard = []
        best_model_name = "RandomForestClassifier"
        best_f1 = -1.0
        best_fitted_model = None
        best_test_preds = None

        for name, clf in models.items():
            train_x = X_train_scaled if name == "LogisticRegression" else X_train
            test_x = X_test_scaled if name == "LogisticRegression" else X_test

            # 5-Fold Stratified Cross-Validation on training split
            try:
                cv_scores = cross_val_score(clf, train_x, y_train, cv=cv, scoring="accuracy")
                cv_mean = round(float(np.mean(cv_scores)), 4)
                cv_std = round(float(np.std(cv_scores)), 4)
            except Exception:
                cv_mean, cv_std = 0.0, 0.0

            clf.fit(train_x, y_train)
            train_preds = clf.predict(train_x)
            test_preds = clf.predict(test_x)

            acc = round(float(accuracy_score(y_test, test_preds)), 4)
            train_acc = round(float(accuracy_score(y_train, train_preds)), 4)
            f1 = round(float(f1_score(y_test, test_preds, average="weighted", zero_division=0)), 4)
            prec = round(float(precision_score(y_test, test_preds, average="weighted", zero_division=0)), 4)
            rec = round(float(recall_score(y_test, test_preds, average="weighted", zero_division=0)), 4)
            overfit_gap = round(train_acc - acc, 4)

            # ROC-AUC calculation
            auc_val = None
            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(test_x)
                    if len(np.unique(y)) == 2:
                        auc_val = round(float(roc_auc_score(y_test, probs[:, 1])), 4)
                    else:
                        auc_val = round(float(roc_auc_score(y_test, probs, multi_class="ovr")), 4)
            except Exception:
                pass

            info = {
                "name": name,
                "cv_5fold_mean": cv_mean,
                "cv_5fold_std": cv_std,
                "test_accuracy": acc,
                "train_accuracy": train_acc,
                "f1_score": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": auc_val if auc_val is not None else "N/A",
                "overfit_gap": overfit_gap
            }
            leaderboard.append(info)

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_fitted_model = clf
                best_test_preds = test_preds

        # Confusion Matrix for Best Model
        conf_mat = confusion_matrix(y_test, best_test_preds).tolist()
        state["confusion_matrix"] = conf_mat

        # Explainable AI: Feature Importances & Directional Impact
        top_features_list = []
        rf_model = models.get("RandomForestClassifier")
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            raw_importances = rf_model.feature_importances_
            total_imp = np.sum(raw_importances)
            norm_imp = (raw_importances / total_imp) if total_imp > 0 else raw_importances
            
            # Compute Directional Impact (correlation with target)
            for idx, f_name in enumerate(feature_names):
                f_vals = X[:, idx]
                corr_val = float(pd.Series(f_vals).corr(pd.Series(y)))
                dir_label = "Positive Risk (+)" if corr_val > 0.05 else ("Protective / Negative (-)" if corr_val < -0.05 else "Neutral")
                top_features_list.append({
                    "feature": f_name,
                    "importance": round(float(norm_imp[idx]), 4),
                    "directional_correlation": round(corr_val, 3) if not np.isnan(corr_val) else 0.0,
                    "impact_direction": dir_label
                })
            top_features_list.sort(key=lambda x: -x["importance"])
            top_features_list = top_features_list[:10]

        best_meta = next(m for m in leaderboard if m["name"] == best_model_name)
        state["model_report"] = {
            "target_column": target_col,
            "problem_type": "classification",
            "model": best_model_name,
            "accuracy": best_meta["test_accuracy"],
            "train_accuracy": best_meta["train_accuracy"],
            "cv_accuracy": f"{best_meta['cv_5fold_mean']} ± {best_meta['cv_5fold_std']}",
            "f1_score": best_meta["f1_score"],
            "precision": best_meta["precision"],
            "recall": best_meta["recall"],
            "roc_auc": best_meta["roc_auc"],
            "overfit_gap": best_meta["overfit_gap"],
            "top_features": [(f["feature"], f["importance"]) for f in top_features_list],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "classes_map": classes_map,
            "is_regularized": regularize_flag
        }
        state["models_evaluated"] = leaderboard
        state["top_features"] = top_features_list

    else:
        # REGRESSION WORKFLOW
        y = np.asarray(y_series.values, dtype=np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg_models = {
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=custom_max_depth, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=80, max_depth=min(custom_max_depth, 4), random_state=42),
            "RidgeRegression": Ridge(alpha=1.0)
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        reg_leaderboard = []
        best_reg_name = "RandomForestRegressor"
        best_r2 = -999.0
        best_reg_model = None

        for name, reg in reg_models.items():
            cv_r2 = cross_val_score(reg, X_train, y_train, cv=cv, scoring="r2")
            reg.fit(X_train, y_train)
            train_preds = reg.predict(X_train)
            test_preds = reg.predict(X_test)

            r2 = round(float(r2_score(y_test, test_preds)), 4)
            train_r2 = round(float(r2_score(y_train, train_preds)), 4)
            rmse = round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 3)
            mae = round(float(mean_absolute_error(y_test, test_preds)), 3)
            gap = round(train_r2 - r2, 4)

            info = {
                "name": name,
                "cv_5fold_mean_r2": round(float(np.mean(cv_r2)), 4),
                "r2_score": r2,
                "train_r2": train_r2,
                "rmse": rmse,
                "mae": mae,
                "overfit_gap": gap
            }
            reg_leaderboard.append(info)

            if r2 > best_r2:
                best_r2 = r2
                best_reg_name = name
                best_reg_model = reg

        rf_reg = reg_models.get("RandomForestRegressor")
        top_features_list = []
        if rf_reg is not None and hasattr(rf_reg, "feature_importances_"):
            raw_imp = rf_reg.feature_importances_
            for idx, f_name in enumerate(feature_names):
                corr_val = float(pd.Series(X[:, idx]).corr(pd.Series(y)))
                top_features_list.append({
                    "feature": f_name,
                    "importance": round(float(raw_imp[idx]), 4),
                    "directional_correlation": round(corr_val, 3) if not np.isnan(corr_val) else 0.0,
                    "impact_direction": "Positive Driver (+)" if corr_val > 0 else "Inverse Driver (-)"
                })
            top_features_list.sort(key=lambda x: -x["importance"])
            top_features_list = top_features_list[:10]

        best_meta = next(m for m in reg_leaderboard if m["name"] == best_reg_name)
        state["model_report"] = {
            "target_column": target_col,
            "problem_type": "regression",
            "model": best_reg_name,
            "accuracy": best_meta["r2_score"],
            "r2": best_meta["r2_score"],
            "train_accuracy": best_meta["train_r2"],
            "rmse": best_meta["rmse"],
            "mae": best_meta["mae"],
            "cv_accuracy": f"{best_meta['cv_5fold_mean_r2']}",
            "overfit_gap": best_meta["overfit_gap"],
            "top_features": [(f["feature"], f["importance"]) for f in top_features_list],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "is_regularized": regularize_flag
        }
        state["models_evaluated"] = reg_leaderboard
        state["top_features"] = top_features_list

    best_m = state["model_report"]["model"]
    acc_val = state["model_report"]["accuracy"]
    gap_val = state["model_report"]["overfit_gap"]
    top_f_str = ", ".join([f["feature"] for f in top_features_list[:3]])

    state.setdefault("agent_trace", []).append(
        f"[ml_agent] Target='{target_col}' ({state['model_report']['problem_type']}) | "
        f"Selected={best_m} (Test Score={acc_val}, Overfit Gap={gap_val}) | "
        f"Top Predictors=[{top_f_str}]"
    )
    return state
