"""
ML Agent — trains a baseline model IF the plan included this step.
Uses a word-boundary heuristic to guess the target column for now; replace
with LLM-based target identification (from user_goal) once API is wired up.
"""
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from agents.state import AgentState

# common target-like column names, checked as a fallback tier before "last column"
COMMON_TARGET_NAMES = {
    "target", "class", "label", "outcome", "status", "result",
    "churn", "y", "diagnosis", "default", "approved",
}


def _guess_target_column(df: pd.DataFrame, goal: str):
    goal_words = set(re.findall(r"[a-z0-9_]+", goal.lower()))

    # ID-like columns are almost never the target -- deprioritize them
    def is_id_like(col: str) -> bool:
        c = col.lower()
        return c.endswith("id") or c == "index" or c.startswith("unnamed")

    # Tier 1: word-overlap match between column name and goal words (word-boundary,
    # so "ca" inside "clinical" no longer false-matches). Pick the column with the
    # MOST overlapping words, not just the first one found, and skip ID-like columns
    # unless nothing else overlaps at all.
    candidates = []
    for col in df.columns:
        if is_id_like(col):
            continue
        col_words = set(re.findall(r"[a-z0-9]+", col.lower()))
        overlap = len(col_words & goal_words)
        if overlap > 0:
            candidates.append((overlap, col))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    # Tier 2: column name is a commonly-used target label
    for col in df.columns:
        if col.lower() in COMMON_TARGET_NAMES:
            return col

    # Tier 3: fallback -- last non-ID column, common convention in toy/Kaggle datasets
    non_id_cols = [c for c in df.columns if not is_id_like(c)]
    return non_id_cols[-1] if non_id_cols else df.columns[-1]


def run(state: AgentState) -> AgentState:
    df = pd.read_csv(state["dataset_path"]).dropna()
    target_col = _guess_target_column(df, state["user_goal"])

    # ID-like columns are dropped from features
    def is_id_like(col: str) -> bool:
        c = col.lower()
        return c.endswith("id") or c == "index" or c.startswith("unnamed")

    feature_cols = [c for c in df.columns if c != target_col and not is_id_like(c)]
    X_df = df[feature_cols].copy()
    y_series = df[target_col].copy()

    # Preprocess categorical features using one-hot encoding
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    if X_df.shape[1] == 0:
        state["model_report"] = {"status": "skipped", "reason": "no features available"}
        return state

    # Convert all boolean columns to float/int
    for col in X_df.columns:
        if X_df[col].dtype == bool:
            X_df[col] = X_df[col].astype(int)

    X = np.asarray(X_df.values, dtype=np.float32)
    feature_names = list(X_df.columns)

    # Check whether target is classification or regression
    is_classification = y_series.nunique() <= 10 or y_series.dtype == object

    if is_classification:
        if not pd.api.types.is_numeric_dtype(y_series):
            le = LabelEncoder()
            y = np.asarray(le.fit_transform(y_series.astype(str)), dtype=np.int32)
        else:
            y = np.asarray(y_series.values, dtype=np.int32)

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        candidate_models = {
            "RandomForestClassifier": (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), X_train, X_test),
            "GradientBoostingClassifier": (GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42), X_train, X_test),
            "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled)
        }

        models_evaluated = []
        best_model_name = "RandomForestClassifier"
        best_f1 = -1.0
        best_clf = None

        for name, (clf, train_x, test_x) in candidate_models.items():
            clf.fit(train_x, y_train)
            train_preds = clf.predict(train_x)
            test_preds = clf.predict(test_x)

            acc = float(accuracy_score(y_test, test_preds))
            train_acc = float(accuracy_score(y_train, train_preds))
            f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
            gap = round(train_acc - acc, 3)

            info = {
                "name": name,
                "accuracy": round(acc, 3),
                "train_accuracy": round(train_acc, 3),
                "f1_score": round(f1, 3),
                "overfit_gap": gap
            }
            models_evaluated.append(info)

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_clf = clf

        # Feature importances from best tree model or RandomForest
        importances_dict = {}
        rf = candidate_models["RandomForestClassifier"]
        if hasattr(rf, "feature_importances_"):
            importances = dict(zip(feature_names, rf.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: -x[1])[:7]
        else:
            top_features = []

        best_meta = next(m for m in models_evaluated if m["name"] == best_model_name)
        state["model_report"] = {
            "target_column": target_col,
            "problem_type": "classification",
            "model": best_model_name,
            "accuracy": best_meta["accuracy"],
            "train_accuracy": best_meta["train_accuracy"],
            "f1_score": best_meta["f1_score"],
            "overfit_gap": best_meta["overfit_gap"],
            "top_features": top_features,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        state["models_evaluated"] = models_evaluated
    else:
        # Regression
        y = np.asarray(y_series.values, dtype=np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_squared_error

        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        train_preds = reg.predict(X_train)

        r2 = round(float(r2_score(y_test, preds)), 3)
        train_r2 = round(float(r2_score(y_train, train_preds)), 3)
        rmse = round(float(mean_squared_error(y_test, preds) ** 0.5), 2)
        gap = round(train_r2 - r2, 3)

        importances = dict(zip(feature_names, reg.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:7]

        state["model_report"] = {
            "target_column": target_col,
            "problem_type": "regression",
            "model": "RandomForestRegressor",
            "accuracy": r2,  # R2
            "f1_score": r2,
            "r2": r2,
            "rmse": rmse,
            "overfit_gap": gap,
            "top_features": top_features,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        state["models_evaluated"] = [{
            "name": "RandomForestRegressor",
            "r2": r2,
            "rmse": rmse,
            "overfit_gap": gap
        }]

    state.setdefault("agent_trace", []).append(
        f"[ml_agent] target={target_col} best_model={state['model_report']['model']} "
        f"score={state['model_report']['accuracy']} overfit_gap={state['model_report'].get('overfit_gap', 0)} "
        f"top_features={[f[0] for f in top_features[:3]]}"
    )
    return state
