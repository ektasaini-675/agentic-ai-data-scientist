"""Data tools: profile, clean, identify the target. Deterministic and tested.

Each tool takes the current dataframe plus plain arguments and returns a
ToolResult whose `summary` is a plain-English sentence (this sentence is what
agents and the claim verifier read).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ads.state import ToolResult
from ads.tools.registry import register

COMMON_TARGET_NAMES: Set[str] = {
    "target", "class", "label", "outcome", "status", "result", "churn", "y",
    "diagnosis", "default", "approved", "exited", "survived", "num",
}


# ---------------------------------------------------------------- helpers ---
def _tokens(name: Any) -> Set[str]:
    """'customerID' -> {'customer','id'}, 'TotalCharges' -> {'total','charges'}."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _is_text_dtype(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ) and not pd.api.types.is_bool_dtype(s)


def _numeric_share(s: pd.Series) -> float:
    """Share of non-blank values in a text column that parse as numbers."""
    vals = s.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) == 0:
        return 0.0
    return float(pd.to_numeric(vals, errors="coerce").notna().mean())


def _is_id_like(name: Any, s: pd.Series, n_rows: int) -> bool:
    low = str(name).lower()
    if "id" in _tokens(name) or low == "index" or low.startswith("unnamed"):
        return True
    if n_rows > 20:
        n_unique = s.nunique(dropna=True)
        if _is_text_dtype(s):
            # near-unique strings (not numbers stored as text) behave like identifiers
            return n_unique >= 0.98 * n_rows and _numeric_share(s) < 0.95
        if pd.api.types.is_integer_dtype(s):
            return n_unique == n_rows and bool(s.is_monotonic_increasing)
    return False


def _missing_count(s: pd.Series) -> int:
    missing = int(s.isna().sum())
    if _is_text_dtype(s):
        blanks = s.dropna().astype(str).str.strip().eq("").sum()
        missing += int(blanks)
    return missing


# --------------------------------------------------------- profile_dataset ---
@register(
    "profile_dataset",
    "Profile the dataset: size, column kinds, missing values, duplicate rows, "
    "identifier-like columns and numbers stored as text. Call this first.",
)
def profile_dataset(df: pd.DataFrame) -> ToolResult:
    n_rows, n_cols = df.shape
    columns: List[Dict[str, Any]] = []
    text_numeric: List[str] = []
    id_like: List[str] = []
    total_missing = 0
    cols_with_missing = 0

    for name in df.columns:
        s = df[name]
        missing = _missing_count(s)
        total_missing += missing
        cols_with_missing += 1 if missing else 0

        if _is_id_like(name, s, n_rows):
            kind = "id_like"
            id_like.append(str(name))
        elif _is_text_dtype(s) and _numeric_share(s) >= 0.95:
            kind = "text_numeric"
            text_numeric.append(str(name))
        elif s.nunique(dropna=True) == 2:
            kind = "binary"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            kind = "numeric"
        else:
            kind = "categorical"

        columns.append(
            {
                "name": str(name),
                "dtype": str(s.dtype),
                "kind": kind,
                "n_missing": missing,
                "pct_missing": round(100.0 * missing / n_rows, 2) if n_rows else 0.0,
                "n_unique": int(s.nunique(dropna=True)),
            }
        )

    n_duplicates = int(df.duplicated().sum())
    parts = [
        f"Dataset has {n_rows:,} rows and {n_cols} columns",
        f"{total_missing:,} missing cell(s) across {cols_with_missing} column(s)",
        f"{n_duplicates} duplicate row(s)",
    ]
    summary = "; ".join(parts) + "."
    if text_numeric:
        summary += f" Numbers stored as text: {', '.join(text_numeric)}."
    if id_like:
        summary += f" Identifier-like columns: {', '.join(id_like)}."

    data = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicates": n_duplicates,
        "total_missing": total_missing,
        "columns": columns,
        "text_numeric_columns": text_numeric,
        "id_like_columns": id_like,
    }
    return ToolResult(ok=True, data=data, summary=summary)


# -------------------------------------------------------------- clean_data ---
@register(
    "clean_data",
    "Clean the dataset: treat blank text as missing, convert numbers stored as text, "
    "drop duplicate rows, drop heavily-missing and identifier columns, impute the rest "
    "(median for numbers, mode for categories). Pass `target` so target rows are never imputed.",
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Target column (never imputed or dropped)."},
            "drop_duplicates": {"type": "boolean"},
            "drop_id_columns": {"type": "boolean"},
            "impute": {"type": "boolean"},
            "max_missing_pct": {"type": "number", "description": "Drop columns missing more than this percent."},
        },
        "required": [],
    },
    cacheable=False,
)
def clean_data(
    df: pd.DataFrame,
    target: Optional[str] = None,
    drop_duplicates: bool = True,
    drop_id_columns: bool = True,
    impute: bool = True,
    max_missing_pct: float = 60.0,
) -> ToolResult:
    out = df.copy()
    rows_before, cols_before = out.shape
    actions: List[str] = []

    text_cols = [c for c in out.columns if _is_text_dtype(out[c])]

    # 1. blank strings are missing values
    blanks_fixed = 0
    for c in text_cols:
        mask = out[c].notna() & out[c].astype(str).str.strip().eq("")
        if mask.any():
            out.loc[mask, c] = np.nan
            blanks_fixed += int(mask.sum())
    if blanks_fixed:
        actions.append(f"treated {blanks_fixed} blank text cell(s) as missing")

    # 2. numbers stored as text -> numeric
    converted: List[str] = []
    for c in text_cols:
        if _numeric_share(out[c]) >= 0.95:
            out[c] = pd.to_numeric(out[c].astype(str).str.strip(), errors="coerce")
            converted.append(str(c))
    if converted:
        actions.append(f"converted {len(converted)} text column(s) to numeric: {', '.join(converted)}")

    # 3. duplicate rows
    n_dups = 0
    if drop_duplicates:
        n_dups = int(out.duplicated().sum())
        if n_dups:
            out = out.drop_duplicates().reset_index(drop=True)
            actions.append(f"dropped {n_dups} duplicate row(s)")

    # 4. rows with a missing target cannot be imputed honestly
    rows_missing_target = 0
    if target and target in out.columns:
        mask = out[target].isna()
        rows_missing_target = int(mask.sum())
        if rows_missing_target:
            out = out.loc[~mask].reset_index(drop=True)
            actions.append(f"dropped {rows_missing_target} row(s) with missing target '{target}'")

    # 5. heavily-missing columns
    dropped_sparse: List[str] = []
    if len(out):
        for c in list(out.columns):
            if c == target:
                continue
            pct = 100.0 * float(out[c].isna().mean())
            if pct > max_missing_pct:
                dropped_sparse.append(str(c))
        if dropped_sparse:
            out = out.drop(columns=dropped_sparse)
            actions.append(f"dropped {len(dropped_sparse)} column(s) missing over {max_missing_pct:g}%: {', '.join(dropped_sparse)}")

    # 6. identifier columns carry no signal
    dropped_ids: List[str] = []
    if drop_id_columns:
        for c in list(out.columns):
            if c != target and _is_id_like(c, out[c], len(out)):
                dropped_ids.append(str(c))
        if dropped_ids:
            out = out.drop(columns=dropped_ids)
            actions.append(f"dropped identifier column(s): {', '.join(dropped_ids)}")

    # 7. impute what is left
    imputed: Dict[str, Dict[str, Any]] = {}
    if impute:
        for c in out.columns:
            n_missing = int(out[c].isna().sum())
            if n_missing == 0 or c == target:
                continue
            if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c]):
                fill = out[c].median()
                method = "median"
                if pd.isna(fill):
                    fill = 0.0
            else:
                mode = out[c].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "Unknown"
                method = "mode"
            out[c] = out[c].fillna(fill)
            imputed[str(c)] = {"count": n_missing, "method": method}
        if imputed:
            total = sum(v["count"] for v in imputed.values())
            actions.append(f"imputed {total} missing value(s) in {len(imputed)} column(s)")

    rows_after, cols_after = out.shape
    summary = (
        f"Cleaned data: rows from {rows_before:,} to {rows_after:,}, "
        f"columns from {cols_before} to {cols_after}. "
        + ("Actions: " + "; ".join(actions) + "." if actions else "No changes were needed.")
    )
    data = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "actions": actions,
        "converted_columns": converted,
        "dropped_id_columns": dropped_ids,
        "dropped_sparse_columns": dropped_sparse,
        "duplicates_dropped": n_dups,
        "imputed": imputed,
    }
    return ToolResult(ok=True, data=data, summary=summary, dataframe=out)


# --------------------------------------------------------- identify_target ---
@register(
    "identify_target",
    "Infer which column is the prediction target from the user's goal, and whether the "
    "problem is classification or regression.",
    parameters={
        "type": "object",
        "properties": {"goal": {"type": "string", "description": "The user's analytical goal."}},
        "required": ["goal"],
    },
)
def identify_target(df: pd.DataFrame, goal: str) -> ToolResult:
    goal_words = {w for w in re.split(r"[^a-z0-9]+", goal.lower()) if w}
    n = len(df)

    cands = []
    for c in df.columns:
        s = df[c]
        if _is_id_like(c, s, n):
            continue
        toks = _tokens(c)
        overlap = len(toks & goal_words)
        is_common = bool(toks & COMMON_TARGET_NAMES)
        small_domain = s.nunique(dropna=True) <= 10
        cands.append((overlap, is_common, small_domain, c))

    if not cands:
        return ToolResult(ok=False, error="no usable (non-identifier) columns to choose a target from")

    overlapping = [x for x in cands if x[0] > 0]
    if overlapping:
        overlapping.sort(key=lambda x: (-x[0], not x[1], not x[2], str(x[3])))
        target = overlapping[0][3]
        method = "goal word overlap"
        confidence = "high"
        ranked = [x[3] for x in overlapping[:3]]
    else:
        common = [x for x in cands if x[1]]
        if common:
            target = common[0][3]
            method = "common target name"
            confidence = "medium"
            ranked = [x[3] for x in common[:3]]
        else:
            target = cands[-1][3]
            method = "last column fallback"
            confidence = "low"
            ranked = [target]

    s = df[target]
    n_unique = int(s.nunique(dropna=True))
    is_classification = n_unique <= 10 or not pd.api.types.is_numeric_dtype(s)
    problem_type = "classification" if is_classification else "regression"

    summary = (
        f"Target column inferred as '{target}' ({problem_type}, {n_unique} distinct values) "
        f"via {method}; confidence {confidence}."
    )
    data = {
        "target": str(target),
        "problem_type": problem_type,
        "n_unique": n_unique,
        "method": method,
        "confidence": confidence,
        "candidates": [str(c) for c in ranked],
    }
    return ToolResult(ok=True, data=data, summary=summary)


# ----------------------------------------------------------- drop_columns ---
@register(
    "drop_columns",
    "Drop one or more columns from the dataset (used to remove target leakage or unwanted columns).",
    parameters={
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of column names to drop.",
            }
        },
        "required": ["columns"],
    },
    cacheable=False,
    agents=("data_agent",),
)
def drop_columns(df: pd.DataFrame, columns: List[str]) -> ToolResult:
    if not isinstance(columns, (list, tuple)):
        columns = [columns]
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return ToolResult(
            ok=True,
            data={"dropped": [], "remaining_columns": list(df.columns)},
            summary="No specified columns found to drop.",
            dataframe=df,
        )
    new_df = df.drop(columns=existing)
    dropped_str = ", ".join(f"'{c}'" for c in existing)
    summary = f"Dropped {len(existing)} column(s): {dropped_str}."
    return ToolResult(
        ok=True,
        data={"dropped": existing, "remaining_columns": list(new_df.columns)},
        summary=summary,
        dataframe=new_df,
    )
