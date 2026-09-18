"""
modules/data_loader.py
Dataset ingestion, validation, and dynamic data profiling engine.
"""

from typing import Tuple, Dict, Any, Optional
import io
import pandas as pd
import numpy as np

def load_dataset(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Reads an uploaded file object (.csv or .xlsx) into a Pandas DataFrame.
    Returns (DataFrame, None) on success or (None, error_message) on failure.
    """
    if uploaded_file is None:
        return None, "No file provided."

    file_name = uploaded_file.name.lower()
    
    try:
        if file_name.endswith(".csv"):
            # Inspect encoding and delimiter flexibility
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            return None, "Unsupported file format. Please upload a .csv or .xlsx file."
        
        if df.empty:
            return None, "The uploaded file contains no data."
        
        # Clean basic column naming artifacts (strip outer whitespace)
        df.columns = [str(col).strip() for col in df.columns]
        return df, None

    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


def compute_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive statistical and structural data-quality profile.
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    # Column-level inventory
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    
    # Try parsing string columns that might actually be dates
    text_or_obj_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    col_summaries = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        inferred_type = str(series.dtype)
        
        col_summaries.append({
            "Column Name": col,
            "Detected Dtype": inferred_type,
            "Missing Values": n_missing,
            "Missing %": round((n_missing / n_rows) * 100, 2),
            "Unique Values": n_unique,
            "Sample Value": str(series.dropna().iloc[0]) if n_unique > 0 else "N/A"
        })

    col_profile_df = pd.DataFrame(col_summaries)

    # Estimate memory footprint
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return {
        "total_rows": n_rows,
        "total_columns": n_cols,
        "total_missing_cells": total_missing,
        "overall_missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "memory_mb": memory_usage_mb,
        "column_profiles": col_profile_df
    }