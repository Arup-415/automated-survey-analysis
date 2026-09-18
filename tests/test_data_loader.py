import io
import pandas as pd
import pytest
from modules.data_loader import load_dataset, compute_dataset_profile

class DummyFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._io = io.BytesIO(data)
    def read(self, *args, **kwargs):
        return self._io.read(*args, **kwargs)
    def seek(self, *args, **kwargs):
        return self._io.seek(*args, **kwargs)

def test_load_valid_csv():
    csv_bytes = b"id,age,dept\n1,25,Sales\n2,30,Eng"
    file_obj = DummyFile("test.csv", csv_bytes)
    df, err = load_dataset(file_obj)
    assert err is None
    assert df is not None
    assert df.shape == (2, 3)

def test_load_unsupported_format():
    txt_bytes = b"just some text"
    file_obj = DummyFile("test.txt", txt_bytes)
    df, err = load_dataset(file_obj)
    assert df is None
    assert "Unsupported file format" in err

def test_compute_profile():
    df = pd.DataFrame({
        "id": [1, 2, 2],
        "score": [10.0, None, 8.5],
        "feedback": ["Great", "Bad", "Bad"]
    })
    profile = compute_dataset_profile(df)
    assert profile["total_rows"] == 3
    assert profile["total_columns"] == 3
    assert profile["duplicate_rows"] == 0  # row 1 & 2 differ by id
    assert profile["total_missing_cells"] == 1
    assert profile["overall_missing_pct"] == pytest.approx(11.11, 0.05)