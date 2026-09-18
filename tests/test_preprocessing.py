import pandas as pd
import numpy as np
from modules.preprocessing import sanitize_text, clean_survey_data
from modules.schema_detector import detect_column_schema

def test_sanitize_text_pii():
    text = "Please reach out to me at test.user@company.com or 555-123-4567. See https://example.com"
    clean = sanitize_text(text)
    assert "[EMAIL]" in clean
    assert "[PHONE]" in clean
    assert "[URL]" in clean
    assert "test.user@company.com" not in clean

def test_sanitize_junk_text():
    assert sanitize_text("N/A") == ""
    assert sanitize_text("asdfgh") == ""
    assert sanitize_text("  none  ") == ""

def test_clean_survey_data_pipeline():
    raw_data = {
        "id": [1, 2, 2, 3],
        "age": [25.0, np.nan, np.nan, 35.0],
        "dept": ["Sales", "Eng", "Eng", None],
        "feedback": [
            "Contact user@org.com",
            "N/A",
            "N/A",
            "Great service overall!"
        ]
    }
    df = pd.DataFrame(raw_data)
    schema_df = detect_column_schema(df)
    
    cleaned_df, summary = clean_survey_data(
        df, schema_df, drop_duplicates=True, impute_numeric="median", impute_categorical="No Response"
    )
    
    # 1 duplicate row removed (row 2 was exact duplicate of row 1)
    assert summary["duplicates_removed"] == 1
    assert len(cleaned_df) == 3
    # Age missing imputed via median (30.0)
    assert cleaned_df["age"].isna().sum() == 0
    # Dept missing filled with 'No Response'
    assert "No Response" in cleaned_df["dept"].values
    # PII redacted
    assert "[EMAIL]" in cleaned_df["feedback"].iloc[0]
