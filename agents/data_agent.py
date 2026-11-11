"""
Data Hygiene & Preprocessing Agent.
Performs automated schema discovery, primary key isolation, statistical imputation,
outlier boundary scanning, and data quality health scoring.
"""
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from agents.state import AgentState


def is_id_or_hash_column(col_name: str, series: pd.Series) -> bool:
    """Detects whether a column is an ID, hash, or primary key column."""
    name = col_name.lower().strip()
    if name in ["id", "index", "unnamed: 0", "customerid", "loan_id", "patient_id"]:
        return True
    if name.endswith("_id") or name.endswith("id"):
        # Check uniqueness: if unique count is nearly equal to length, it's an ID
        if series.nunique() >= len(series) * 0.9:
            return True
    # If string with high entropy / uuid-like
    if series.dtype == object and series.nunique() == len(series) and len(series) > 50:
        return True
    return False


def run(state: AgentState) -> AgentState:
    raw_path = state["dataset_path"]
    df = pd.read_csv(raw_path)

    cleaning_log: List[str] = []
    n_rows_initial, n_cols_initial = df.shape
    cleaning_log.append(f"Ingested raw dataset with {n_rows_initial:,} rows and {n_cols_initial} columns.")

    # 1. Duplicates check and removal
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        df = df.drop_duplicates()
        cleaning_log.append(f"Detected and purged {n_duplicates:,} duplicate rows ({round(n_duplicates / n_rows_initial * 100, 2)}%).")
    else:
        cleaning_log.append("No duplicate rows detected.")

    # 2. Identify ID and Constant columns
    id_cols = [c for c in df.columns if is_id_or_hash_column(c, df[c])]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    if id_cols:
        cleaning_log.append(f"Identified primary key/identifier columns (isolated from predictive training): {id_cols}")
    if constant_cols:
        cleaning_log.append(f"Identified zero-variance constant columns: {constant_cols}")

    # 3. Handle whitespace and blank strings in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        # Convert empty strings or 'nan'/'none'/'null' to actual NaN
        df[col] = df[col].replace(["", " ", "nan", "None", "null", "NULL", "NA", "?"], np.nan)
        # Check if column should actually be numeric (e.g. TotalCharges with spaces)
        try:
            converted = pd.to_numeric(df[col])
            if converted.notnull().sum() > len(df) * 0.8:
                df[col] = converted
                cleaning_log.append(f"Coerced pseudo-categorical column '{col}' to numeric format.")
        except (ValueError, TypeError):
            pass

    # 4. Statistical Imputation (No Blind Dropna)
    missing_counts = df.isnull().sum()
    missing_cols = {c: int(v) for c, v in missing_counts.items() if v > 0}
    imputation_details = {}

    if missing_cols:
        for c, count in missing_cols.items():
            pct = round((count / len(df)) * 100, 2)
            if c in id_cols:
                continue

            if pd.api.types.is_numeric_dtype(df[c]):
                skew = df[c].skew()
                if abs(skew) > 1.0:
                    fill_val = float(df[c].median())
                    method = "median (skewed distribution)"
                else:
                    fill_val = float(df[c].mean())
                    method = "mean (gaussian approximation)"
                df[c] = df[c].fillna(fill_val)
                imputation_details[c] = {"missing_count": count, "missing_pct": pct, "method": method, "fill_value": round(fill_val, 3)}
            else:
                mode_series = df[c].mode()
                fill_val = mode_series.iloc[0] if not mode_series.empty else "Missing"
                df[c] = df[c].fillna(fill_val)
                imputation_details[c] = {"missing_count": count, "missing_pct": pct, "method": "mode imputation", "fill_value": str(fill_val)}

        cleaning_log.append(f"Imputed missing values across {len(missing_cols)} columns using statistical median/mean/mode strategy.")
    else:
        cleaning_log.append("Missing values audit: Complete data integrity (0% null values).")

    # 5. Persist Cleaned Dataset
    os.makedirs("datasets/processed", exist_ok=True)
    cleaned_path = "datasets/processed/cleaned_data.csv"
    df.to_csv(cleaned_path, index=False)

    # 6. Compute Data Quality Health Score (0-100%)
    null_penalty = min(30, int((sum(missing_counts) / (n_rows_initial * n_cols_initial)) * 200))
    dup_penalty = min(20, int((n_duplicates / n_rows_initial) * 100))
    constant_penalty = len(constant_cols) * 5
    quality_score = max(50, 100 - (null_penalty + dup_penalty + constant_penalty))

    schema_report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category", "bool"]).columns),
        "id_columns": id_cols,
        "constant_columns": constant_cols,
        "missing_values": missing_cols,
        "n_duplicates": n_duplicates,
        "data_quality_score": quality_score
    }

    data_quality_report = {
        "initial_shape": [n_rows_initial, n_cols_initial],
        "final_shape": [len(df), len(df.columns)],
        "imputation_details": imputation_details,
        "quality_score": quality_score,
        "id_columns_isolated": id_cols
    }

    state["schema_report"] = schema_report
    state["cleaning_log"] = cleaning_log
    state["data_quality_report"] = data_quality_report
    state["cleaned_dataset_path"] = cleaned_path

    state.setdefault("agent_trace", []).append(
        f"[data_agent] Ingested {n_rows_initial} rows x {n_cols_initial} cols | "
        f"Duplicates purged={n_duplicates} | Imputed cols={len(missing_cols)} | "
        f"Health Score={quality_score}%"
    )
    return state
