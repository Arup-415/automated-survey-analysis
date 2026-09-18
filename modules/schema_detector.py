"""
modules/schema_detector.py
Dynamic column type and survey role inference engine.
"""

from typing import Dict, Any, List
import re
import pandas as pd
import numpy as np

NPS_KEYWORDS = [r"nps", r"recommend", r"promoter"]
CSAT_KEYWORDS = [r"csat", r"satisfaction", r"satisfied", r"rating"]
CES_KEYWORDS = [r"ces", r"effort", r"easy", r"ease"]
ID_KEYWORDS = [r"^id$", r"_id$", r"id_", r"response.*id", r"resp.*id", r"respondent", r"participant"]
DATE_KEYWORDS = [r"date", r"time", r"timestamp", r"submitted", r"created"]
DEMOGRAPHIC_KEYWORDS = [r"age", r"gender", r"tenure", r"income", r"education", r"years"]
GROUPING_KEYWORDS = [r"dept", r"department", r"region", r"country", r"location", r"team", r"division"]
TEXT_KEYWORDS = [r"comment", r"feedback", r"explain", r"thought", r"text", r"why", r"opinion", r"notes"]

def _matches_any(pattern_list: List[str], text: str) -> bool:
    clean_text = str(text).lower().strip()
    return any(re.search(pat, clean_text) is not None for pat in pattern_list)

def detect_column_schema(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    results = []

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        n_unique = non_null.nunique()
        col_lower = str(col).lower().strip()

        # 1. Null columns
        if len(non_null) == 0:
            results.append({
                "Column Name": col,
                "Detected Type": "Empty",
                "Survey Role": "Unusable",
                "Confidence": "High",
                "Details": "Contains 100% missing values."
            })
            continue

        # 2. DateTime
        is_date = False
        if pd.api.types.is_datetime64_any_dtype(series):
            is_date = True
        elif _matches_any(DATE_KEYWORDS, col_lower):
            try:
                pd.to_datetime(non_null.iloc[:min(30, len(non_null))], errors="raise")
                is_date = True
            except Exception:
                is_date = False

        if is_date:
            results.append({
                "Column Name": col,
                "Detected Type": "DateTime",
                "Survey Role": "Timestamp",
                "Confidence": "High" if _matches_any(DATE_KEYWORDS, col_lower) else "Medium",
                "Details": "Temporal timestamp data"
            })
            continue

        # 3. Respondent ID
        if _matches_any(ID_KEYWORDS, col_lower) and (n_unique / max(n_rows, 1) > 0.50):
            results.append({
                "Column Name": col,
                "Detected Type": "Identifier",
                "Survey Role": "Respondent ID",
                "Confidence": "High",
                "Details": "Unique respondent identifier"
            })
            continue

        # 4. Numeric / Ordinal Metrics
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if not is_numeric:
            try:
                converted = pd.to_numeric(non_null, errors="raise")
                is_numeric = True
                series_vals = converted
            except Exception:
                is_numeric = False

        if is_numeric:
            vals = pd.to_numeric(non_null, errors="coerce").dropna()
            min_val = float(vals.min())
            max_val = float(vals.max())
            is_discrete = bool(np.all(np.equal(np.mod(vals.values, 1), 0)))

            # 4a. Binary
            if is_discrete and n_unique == 2 and set(vals.unique()).issubset({0, 1}):
                results.append({
                    "Column Name": col,
                    "Detected Type": "Binary",
                    "Survey Role": "Binary Flag",
                    "Confidence": "High",
                    "Details": "Binary 0/1"
                })
                continue

            # 4b. CSAT (1 to 5 scale)
            if is_discrete and min_val >= 1 and max_val <= 5:
                is_kw = _matches_any(CSAT_KEYWORDS, col_lower)
                results.append({
                    "Column Name": col,
                    "Detected Type": "Ordinal",
                    "Survey Role": "CSAT Question" if is_kw else "Likert Scale (1-5)",
                    "Confidence": "High" if is_kw else "Medium",
                    "Details": f"1-5 Scale [{int(min_val)}, {int(max_val)}]"
                })
                continue

            # 4c. CES (1 to 7 scale)
            if is_discrete and min_val >= 1 and max_val <= 7:
                is_kw = _matches_any(CES_KEYWORDS, col_lower)
                results.append({
                    "Column Name": col,
                    "Detected Type": "Ordinal",
                    "Survey Role": "CES Question" if is_kw else "Likert Scale (1-7)",
                    "Confidence": "High" if is_kw else "Medium",
                    "Details": f"1-7 Scale [{int(min_val)}, {int(max_val)}]"
                })
                continue

            # 4d. NPS (0 to 10 scale)
            if is_discrete and min_val >= 0 and max_val <= 10:
                is_kw = _matches_any(NPS_KEYWORDS, col_lower)
                results.append({
                    "Column Name": col,
                    "Detected Type": "Ordinal",
                    "Survey Role": "NPS Question" if is_kw else "Scale / Rating (0-10)",
                    "Confidence": "High" if is_kw else "Medium",
                    "Details": f"0-10 Scale [{int(min_val)}, {int(max_val)}]"
                })
                continue

            # 4e. Demographic Numeric
            if _matches_any(DEMOGRAPHIC_KEYWORDS, col_lower):
                results.append({
                    "Column Name": col,
                    "Detected Type": "Numeric",
                    "Survey Role": "Demographic Variable",
                    "Confidence": "High",
                    "Details": f"Numeric [{min_val}, {max_val}]"
                })
                continue

            # 4f. Continuous Metric
            results.append({
                "Column Name": col,
                "Detected Type": "Numeric",
                "Survey Role": "Continuous Metric",
                "Confidence": "Medium",
                "Details": f"Numeric [{min_val}, {max_val}]"
            })
            continue

        # 5. String / Text / Categorical
        str_series = non_null.astype(str)
        avg_words = float(str_series.str.split().str.len().mean())
        avg_chars = float(str_series.str.len().mean())
        is_text_name = _matches_any(TEXT_KEYWORDS, col_lower)

        # 5a. Open-ended Text (keyword match OR length heuristic)
        if is_text_name or avg_words > 3.0 or avg_chars > 20:
            results.append({
                "Column Name": col,
                "Detected Type": "Text",
                "Survey Role": "Open-ended Response",
                "Confidence": "High" if is_text_name else "Medium",
                "Details": f"Avg Words: {avg_words:.1f}"
            })
            continue

        # 5b. Binary Categorical
        if n_unique == 2:
            results.append({
                "Column Name": col,
                "Detected Type": "Binary",
                "Survey Role": "Binary Flag",
                "Confidence": "High",
                "Details": "2 Categories"
            })
            continue

        # 5c. Demographic Categorical
        if _matches_any(DEMOGRAPHIC_KEYWORDS, col_lower):
            results.append({
                "Column Name": col,
                "Detected Type": "Categorical",
                "Survey Role": "Demographic Variable",
                "Confidence": "High",
                "Details": f"Demographic ({n_unique} unique)"
            })
            continue

        # 5d. Grouping
        if _matches_any(GROUPING_KEYWORDS, col_lower) or (n_unique <= 30 and n_rows > 30):
            results.append({
                "Column Name": col,
                "Detected Type": "Categorical",
                "Survey Role": "Grouping Variable",
                "Confidence": "High" if _matches_any(GROUPING_KEYWORDS, col_lower) else "Medium",
                "Details": f"Groups ({n_unique} unique)"
            })
            continue

        # 5e. Fallback
        results.append({
            "Column Name": col,
            "Detected Type": "Categorical",
            "Survey Role": "General Categorical",
            "Confidence": "Low",
            "Details": f"Unique: {n_unique}"
        })

    return pd.DataFrame(results)

def get_role_mapping(schema_df: pd.DataFrame) -> Dict[str, Any]:
    nps_cols = schema_df[schema_df["Survey Role"] == "NPS Question"]["Column Name"].tolist()
    csat_cols = schema_df[schema_df["Survey Role"] == "CSAT Question"]["Column Name"].tolist()
    if not csat_cols:
        csat_cols = schema_df[schema_df["Survey Role"] == "Likert Scale (1-5)"]["Column Name"].tolist()

    ces_cols = schema_df[schema_df["Survey Role"] == "CES Question"]["Column Name"].tolist()
    text_cols = schema_df[schema_df["Detected Type"] == "Text"]["Column Name"].tolist()
    cat_cols = schema_df[schema_df["Detected Type"].isin(["Categorical", "Binary"])]["Column Name"].tolist()
    num_cols = schema_df[schema_df["Detected Type"].isin(["Numeric", "Ordinal"])]["Column Name"].tolist()

    return {
        "nps_col": nps_cols[0] if nps_cols else None,
        "csat_col": csat_cols[0] if csat_cols else None,
        "ces_col": ces_cols[0] if ces_cols else None,
        "text_cols": text_cols,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols
    }
