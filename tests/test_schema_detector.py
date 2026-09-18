import pandas as pd
from modules.schema_detector import detect_column_schema, get_role_mapping

def test_schema_detection_roles():
    data = {
        "resp_id": [f"ID_{i}" for i in range(100)],
        "recommend_score": [i % 11 for i in range(100)],
        "satisfaction_rating": [(i % 5) + 1 for i in range(100)],
        "department_name": ["Sales", "Eng", "HR", "Product"] * 25,
        "user_feedback": ["This is a reasonably long comment providing survey feedback about the tool." for _ in range(100)],
        "age_years": [20 + (i % 40) for i in range(100)]
    }
    df = pd.DataFrame(data)
    schema_df = detect_column_schema(df)
    mapping = get_role_mapping(schema_df)

    assert mapping["nps_col"] == "recommend_score"
    assert mapping["csat_col"] == "satisfaction_rating"
    assert "user_feedback" in mapping["text_cols"]
    assert "department_name" in mapping["categorical_cols"]
    assert "age_years" in mapping["numeric_cols"]

def test_missing_metrics_fallback():
    data = {
        "random_numbers": [10.5, 20.3, 30.1],
        "city": ["Paris", "Tokyo", "London"]
    }
    df = pd.DataFrame(data)
    schema_df = detect_column_schema(df)
    mapping = get_role_mapping(schema_df)

    assert mapping["nps_col"] is None
    assert mapping["csat_col"] is None
    assert len(mapping["text_cols"]) == 0
