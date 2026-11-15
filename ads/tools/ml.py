"""ML tools: honest model training, permutation importance, leakage detection.

Fixes over the prototype:
  - Models are chosen by stratified cross-validation on the TRAIN split only;
    the held-out test set is touched once, at the end (the prototype picked
    the best model using the test set, which inflates scores).
  - ROC-AUC / F1 are reported next to accuracy, plus the majority-class
    baseline, so class imbalance cannot hide behind accuracy.
  - Importance is permutation importance on held-out data, computed per
    ORIGINAL column (not per one-hot dummy).
  - `detect_leakage` flags features that predict the target almost perfectly
    on their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ads.config import SETTINGS
from ads.context import RunContext
from ads.state import ToolResult
from ads.tools.data import _is_id_like, _tokens
from ads.tools.registry import register
from ads.tools.stats import is_classification_target, is_numeric, positive_class

MAX_CATEGORIES = 50


# --------------------------------------------------------------- modelling ---
@dataclass
class _Fit:
    pipe: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    clf: bool
    binary: bool
    report: Dict[str, Any]


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if is_numeric(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ]
    )


def _estimators(clf: bool) -> Dict[str, Any]:
    seed = SETTINGS.random_seed
    if clf:
        return {
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(
                n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
            ),
        }
    return {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    }


def _evaluate(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, clf: bool, binary: bool) -> Dict[str, Optional[float]]:
    if not clf:
        pred = pipe.predict(X)
        return {
            "r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        }
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    try:
        auc = float(roc_auc_score(y, proba[:, 1])) if binary else float(
            roc_auc_score(y, proba, multi_class="ovr", average="weighted")
        )
    except ValueError:
        auc = None
    return {
        "roc_auc": auc,
        "f1": float(f1_score(y, pred)) if binary else float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def _fit_best(ctx: RunContext, df: pd.DataFrame, target: str, exclude: Sequence[str] = (), cv_folds: int = 5) -> _Fit:
    key = ("fit", target, tuple(sorted(exclude)), ctx.dataset_version, int(cv_folds))
    if key in ctx.cache:
        return ctx.cache[key]
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found")

    seed = SETTINGS.random_seed
    d = df[df[target].notna()].reset_index(drop=True)
    y_raw = d[target]
    clf = is_classification_target(y_raw)

    drop = set(exclude) | {target}
    dropped_id, dropped_card = [], []
    for c in d.columns:
        if c in drop:
            continue
        if _is_id_like(c, d[c], len(d)):
            drop.add(c)
            dropped_id.append(str(c))
        elif not is_numeric(d[c]) and d[c].nunique(dropna=True) > MAX_CATEGORIES:
            drop.add(c)
            dropped_card.append(str(c))
    X = d.drop(columns=[c for c in drop if c in d.columns])
    if X.shape[1] == 0:
        raise ValueError("no usable feature columns")
    for c in X.columns:
        if not is_numeric(X[c]):
            s = X[c]
            X[c] = s.astype(str).where(s.notna(), np.nan)

    pos = None
    binary = False
    if clf:
        pos = positive_class(y_raw)
        binary = pos is not None
        y = (y_raw == pos).astype(int).to_numpy() if binary else pd.factorize(y_raw)[0]
        counts = np.bincount(y)
        n_min = int(counts.min())
        if n_min < 2:
            raise ValueError("a class has fewer than 2 rows; cannot cross-validate")
        folds = min(int(cv_folds), int(n_min * 0.8))
        if folds < 2:
            raise ValueError("too few rows in the smallest class for cross-validation")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        if binary:
            scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}
        else:
            scoring = {
                "roc_auc": "roc_auc_ovr_weighted",
                "f1": "f1_macro",
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
            }
        select = "roc_auc"
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        folds = min(int(cv_folds), max(2, len(d) // 20))
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
        select = "r2"

    models: Dict[str, Dict[str, Dict[str, float]]] = {}
    fitted: Dict[str, Pipeline] = {}
    for name, est in _estimators(clf).items():
        pipe = Pipeline([("prep", _preprocessor(X_tr)), ("model", est)])
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        entry: Dict[str, Dict[str, float]] = {}
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            if metric in ("mae", "rmse"):
                vals = -vals
            entry[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        models[name] = entry
        fitted[name] = pipe

    best = max(models, key=lambda m: models[m][select]["mean"])
    best_pipe = fitted[best].fit(X_tr, y_tr)
    test = _evaluate(best_pipe, X_te, y_te, clf, binary)
    train = _evaluate(best_pipe, X_tr, y_tr, clf, binary)
    gap = None
    if train.get(select) is not None:
        gap = float(train[select] - models[best][select]["mean"])

    if clf:
        baseline = {"accuracy": float(np.bincount(y_te).max() / len(y_te))}
    else:
        baseline = {"mae": float(np.mean(np.abs(y_te - np.mean(y_tr))))}

    report: Dict[str, Any] = {
        "target": target,
        "problem_type": "classification" if clf else "regression",
        "positive_class": None if pos is None else str(pos),
        "minority_share": float(np.bincount(y).min() / len(y)) if clf else None,
        "n_rows_used": int(len(d)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(X.shape[1]),
        "features": [str(c) for c in X.columns],
        "features_dropped": {"id_like": dropped_id, "high_cardinality": dropped_card, "excluded": sorted(exclude)},
        "cv_folds": int(folds),
        "selection_metric": select,
        "models": models,
        "best_model": best,
        "cv_best": {m: v["mean"] for m, v in models[best].items()},
        "cv_best_std": {m: v["std"] for m, v in models[best].items()},
        "train": train,
        "test": test,
        "train_vs_cv_gap": gap,
        "baseline": baseline,
    }
    fit = _Fit(best_pipe, X_tr, X_te, y_tr, y_te, clf, binary, report)
    ctx.cache[key] = fit
    return fit


def _importance(
    ctx: RunContext, df: pd.DataFrame, target: str, exclude: Sequence[str] = (), n_repeats: int = 5
) -> Dict[str, Any]:
    key = ("perm", target, tuple(sorted(exclude)), ctx.dataset_version, int(n_repeats))
    if key in ctx.cache:
        return ctx.cache[key]
    fit = _fit_best(ctx, df, target, exclude)
    if not fit.clf:
        scoring, metric = "r2", "R2"
    elif fit.binary:
        scoring, metric = "roc_auc", "ROC-AUC"
    else:
        scoring, metric = "roc_auc_ovr_weighted", "ROC-AUC"
    res = sk_permutation_importance(
        fit.pipe,
        fit.X_test,
        fit.y_test,
        scoring=scoring,
        n_repeats=int(n_repeats),
        random_state=SETTINGS.random_seed,
        n_jobs=1,
    )
    ranked = sorted(
        (
            {
                "feature": str(f),
                "importance_mean": float(m),
                "importance_std": float(s),
            }
            for f, m, s in zip(fit.X_test.columns, res.importances_mean, res.importances_std)
        ),
        key=lambda r: -r["importance_mean"],
    )
    out = {"metric": metric, "model": fit.report["best_model"], "features": ranked}
    ctx.cache[key] = out
    return out


# -------------------------------------------------------------------- tools ---
_TARGET_PARAMS = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "exclude": {"type": "array", "items": {"type": "string"}, "description": "Columns to leave out (e.g. leakage)."},
    },
    "required": ["target"],
}


@register(
    "train_cv",
    "Train and compare models with stratified cross-validation on a train split, then score the best "
    "model once on a held-out test set. Reports ROC-AUC/F1/accuracy (or R2/MAE) and a majority-class baseline. "
    "Use `exclude` to drop suspected leakage columns.",
    parameters={
        "type": "object",
        "properties": {
            **_TARGET_PARAMS["properties"],
            "cv_folds": {"type": "integer", "description": "Default 5."},
        },
        "required": ["target"],
    },
    needs_ctx=True,
)
def train_cv(
    df: pd.DataFrame,
    target: str,
    exclude: Optional[List[str]] = None,
    cv_folds: int = 5,
    ctx: Optional[RunContext] = None,
) -> ToolResult:
    fit = _fit_best(ctx, df, target, tuple(exclude or ()), int(cv_folds))
    r = fit.report
    cv, sd, te = r["cv_best"], r["cv_best_std"], r["test"]
    if r["problem_type"] == "classification":
        summary = (
            f"Best model {r['best_model']}, cross-validated on {r['n_train']:,} training rows "
            f"({r['cv_folds']} folds): ROC-AUC {cv['roc_auc']:.2f} +/- {sd['roc_auc']:.2f}, F1 {cv['f1']:.2f}. "
            f"On {r['n_test']:,} held-out rows: ROC-AUC {te['roc_auc'] if te['roc_auc'] is not None else float('nan'):.2f}, "
            f"F1 {te['f1']:.2f}, accuracy {te['accuracy']:.2f} versus {r['baseline']['accuracy']:.2f} "
            f"for always predicting the majority class."
        )
        if r["positive_class"] is not None and r["minority_share"] is not None and r["minority_share"] < 0.2:
            summary += f" Positive class '{r['positive_class']}' is only {r['minority_share']:.1%} of rows."
    else:
        summary = (
            f"Best model {r['best_model']}, cross-validated on {r['n_train']:,} training rows "
            f"({r['cv_folds']} folds): R2 {cv['r2']:.2f} +/- {sd['r2']:.2f}, MAE {cv['mae']:.2f}. "
            f"On {r['n_test']:,} held-out rows: R2 {te['r2']:.2f}, MAE {te['mae']:.2f} "
            f"versus {r['baseline']['mae']:.2f} for always predicting the training mean."
        )
    return ToolResult(ok=True, data=r, summary=summary)


@register(
    "permutation_importance",
    "Which original features matter most: the drop in held-out score when each feature is shuffled. "
    "Trains the best model first if needed.",
    parameters={
        "type": "object",
        "properties": {**_TARGET_PARAMS["properties"], "n_repeats": {"type": "integer", "description": "Default 5."}},
        "required": ["target"],
    },
    needs_ctx=True,
)
def permutation_importance(
    df: pd.DataFrame,
    target: str,
    exclude: Optional[List[str]] = None,
    n_repeats: int = 5,
    ctx: Optional[RunContext] = None,
) -> ToolResult:
    imp = _importance(ctx, df, target, tuple(exclude or ()), int(n_repeats))
    top = imp["features"][:3]
    parts = [f"{f['feature']} ({f['importance_mean']:.3f})" for f in top]
    summary = (
        f"Top predictors by permutation importance on held-out data (drop in {imp['metric']} when shuffled, "
        f"model {imp['model']}): " + ", ".join(parts) + "."
    )
    data = {
        "target": target,
        "metric": imp["metric"],
        "model": imp["model"],
        "features": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()} for f in imp["features"]
        ],
    }
    return ToolResult(ok=True, data=data, summary=summary)


@register(
    "detect_leakage",
    "Flag target leakage: features that predict the target almost perfectly on their own "
    "(single-feature cross-validated score above a threshold), or whose name echoes the target. "
    "Run this BEFORE trusting any model score.",
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "threshold": {"type": "number", "description": "Default 0.95."},
        },
        "required": ["target"],
    },
)
def detect_leakage(df: pd.DataFrame, target: str, threshold: float = 0.95, max_features: int = 60) -> ToolResult:
    if target not in df.columns:
        return ToolResult(ok=False, error=f"target column '{target}' not found")
    seed = SETTINGS.random_seed
    d = df[df[target].notna()].reset_index(drop=True)
    y_raw = d[target]
    clf = is_classification_target(y_raw)

    if clf:
        y = pd.factorize(y_raw)[0]
        counts = np.bincount(y)
        if len(counts) < 2:
            return ToolResult(ok=False, error="target has a single class")
        k = min(3, int(counts.min()))
        if k < 2:
            return ToolResult(ok=False, error="too few rows in the smallest class for leakage checks")
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        scoring = "roc_auc" if len(counts) == 2 else "roc_auc_ovr_weighted"
        metric = "ROC-AUC"
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        scoring, metric = "r2", "R2"

    target_tokens = _tokens(target)
    rows: List[Dict[str, Any]] = []
    for c in [c for c in d.columns if c != target][:max_features]:
        s = d[c]
        if s.nunique(dropna=True) < 2:
            continue
        if is_numeric(s):
            x = s.fillna(s.median()).to_numpy(dtype=float).reshape(-1, 1)
        else:
            x = pd.factorize(s.astype(str))[0].reshape(-1, 1).astype(float)
        model = (
            DecisionTreeClassifier(max_depth=3, random_state=seed)
            if clf
            else DecisionTreeRegressor(max_depth=3, random_state=seed)
        )
        try:
            score = float(np.mean(cross_val_score(model, x, y, cv=cv, scoring=scoring)))
        except Exception:
            continue
        if not np.isfinite(score):
            continue
        name_overlap = bool(_tokens(c) & target_tokens)
        rows.append(
            {
                "feature": str(c),
                "score": round(score, 4),
                "name_overlap": name_overlap,
                "suspect": bool(score >= threshold or (name_overlap and score >= 0.8)),
            }
        )

    rows.sort(key=lambda r: -r["score"])
    suspects = [r for r in rows if r["suspect"]]
    head = f"Leakage check on {len(rows)} feature(s) (single-feature {metric}, threshold {threshold:.2f})"
    if suspects:
        parts = [f"'{r['feature']}' ({r['score']:.2f})" for r in suspects[:5]]
        summary = f"{head}: {len(suspects)} suspect feature(s): " + ", ".join(parts) + "."
    elif rows:
        summary = (
            f"{head}: no feature predicts the target alone above the threshold; "
            f"the highest is '{rows[0]['feature']}' at {rows[0]['score']:.2f}."
        )
    else:
        summary = f"{head}: no feature could be scored."

    data = {
        "target": target,
        "metric": metric,
        "threshold": threshold,
        "n_features_checked": len(rows),
        "features": rows,
        "suspects": [r["feature"] for r in suspects],
    }
    return ToolResult(ok=True, data=data, summary=summary)
