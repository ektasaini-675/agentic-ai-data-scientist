"""
ML Agent — trains a baseline model IF the plan included this step.
Uses a simple heuristic to guess the target column for now; replace with
LLM-based target identification (from user_goal) once API is wired up.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from agents.state import AgentState


def _guess_target_column(df: pd.DataFrame, goal: str):
    goal = goal.lower()
    for col in df.columns:
        if col.lower() in goal:
            return col
    # fallback: last column, common convention in toy datasets
    return df.columns[-1]


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
