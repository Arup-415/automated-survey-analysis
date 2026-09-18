"""
modules/preprocessing.py
Data cleansing, missing value imputation (Mean, Median, Mode, Interpolation, FFill, BFill, Drop),
duplicate handling, and PII masking.
"""

from typing import Tuple, Dict, Any, Union
import re
import pandas as pd
import numpy as np

EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
URL_REGEX = r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_+.~#?&/=]*"

JUNK_RESPONSES = {"n/a", "na", "none", "nil", "asdf", "asdfgh", "no", "nothing", "-", ".", "..."}

def sanitize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    val = str(text).strip()
    if val.lower() in JUNK_RESPONSES:
        return ""
    val = re.sub(EMAIL_REGEX, "[EMAIL]", val)
    val = re.sub(PHONE_REGEX, "[PHONE]", val)
    val = re.sub(URL_REGEX, "[URL]", val)
    val = re.sub(r"\s+", " ", val)
    return val

def clean_survey_data(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    drop_duplicates: bool = True,
    impute_numeric: str = "median",
    impute_categorical: str = "No Response",
    categorical_constant: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputation strategies supported:
    - Numeric: "median", "mean", "mode", "interpolate", "ffill", "bfill", "drop_rows", "none"
    - Categorical: "mode", "ffill", "bfill", "drop_rows", "none", or any constant string fill value.
    """
    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)
    
    # 1. Deduplication
    dup_count = int(cleaned_df.duplicated().sum())
    if drop_duplicates and dup_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    text_cols = schema_df[schema_df["Detected Type"] == "Text"]["Column Name"].tolist()
    num_cols = schema_df[schema_df["Detected Type"].isin(["Numeric", "Ordinal"])]["Column Name"].tolist()
    cat_cols = schema_df[schema_df["Detected Type"].isin(["Categorical", "Binary"])]["Column Name"].tolist()

    pii_counts = {col: 0 for col in text_cols}
    junk_counts = {col: 0 for col in text_cols}

    # 2. Text Normalization & PII Sanitization
    for col in text_cols:
        if col in cleaned_df.columns:
            def check_pii(val):
                if pd.isna(val):
                    return False
                s = str(val)
                return bool(re.search(EMAIL_REGEX, s) or re.search(PHONE_REGEX, s) or re.search(URL_REGEX, s))

            def check_junk(val):
                if pd.isna(val):
                    return False
                return str(val).lower().strip() in JUNK_RESPONSES

            pii_counts[col] = int(cleaned_df[col].apply(check_pii).sum())
            junk_counts[col] = int(cleaned_df[col].apply(check_junk).sum())
            cleaned_df[col] = cleaned_df[col].apply(sanitize_text)
            cleaned_df[col] = cleaned_df[col].replace("", np.nan)

    # 3. Numeric Imputation
    for col in num_cols:
        if col in cleaned_df.columns and cleaned_df[col].isna().sum() > 0:
            if impute_numeric == "median":
                med_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(med_val)
            elif impute_numeric == "mean":
                mean_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(mean_val)
            elif impute_numeric == "mode":
                mode_series = cleaned_df[col].mode()
                if not mode_series.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_series.iloc[0])
            elif impute_numeric == "interpolate":
                cleaned_df[col] = cleaned_df[col].interpolate(method="linear").bfill().ffill()
            elif impute_numeric == "ffill":
                cleaned_df[col] = cleaned_df[col].ffill().bfill()
            elif impute_numeric == "bfill":
                cleaned_df[col] = cleaned_df[col].bfill().ffill()
            elif impute_numeric == "drop_rows":
                cleaned_df = cleaned_df.dropna(subset=[col])

    # 4. Categorical Imputation
    cat_method = str(impute_categorical).lower().strip()
    fill_constant = categorical_constant if categorical_constant is not None else impute_categorical

    for col in cat_cols:
        if col in cleaned_df.columns and cleaned_df[col].isna().sum() > 0:
            if cat_method == "mode":
                mode_series = cleaned_df[col].mode()
                if not mode_series.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_series.iloc[0])
            elif cat_method == "ffill":
                cleaned_df[col] = cleaned_df[col].ffill().bfill()
            elif cat_method == "bfill":
                cleaned_df[col] = cleaned_df[col].bfill().ffill()
            elif cat_method == "drop_rows":
                cleaned_df = cleaned_df.dropna(subset=[col])
            elif cat_method == "none":
                pass
            else:
                cleaned_df[col] = cleaned_df[col].astype(object).fillna(str(fill_constant))

    cleaned_df = cleaned_df.reset_index(drop=True)

    summary = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned_df),
        "duplicates_removed": dup_count if drop_duplicates else 0,
        "pii_redactions": pii_counts,
        "junk_filtered": junk_counts
    }
    return cleaned_df, summary
