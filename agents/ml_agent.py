"""
ML Agent — trains a baseline model IF the plan included this step.
Uses a word-boundary heuristic to guess the target column for now; replace
with LLM-based target identification (from user_goal) once API is wired up.
"""
import re
import pandas as pd
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

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # encode any non-numeric columns crudely for this baseline
    X = X.select_dtypes(include="number").fillna(0)
    if X.shape[1] == 0:
        state["model_report"] = {"status": "skipped", "reason": "no numeric features available"}
        return state

    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    importances = dict(zip(X.columns, clf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

    state["model_report"] = {
        "target_column": target_col,
        "model": "RandomForestClassifier (baseline)",
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "f1_score": round(f1_score(y_test, preds, average="weighted"), 3),
        "top_features": top_features,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    state.setdefault("agent_trace", []).append(
        f"[ml_agent] target={target_col} acc={state['model_report']['accuracy']} "
        f"top_features={[f[0] for f in top_features]}"
    )
    return state
