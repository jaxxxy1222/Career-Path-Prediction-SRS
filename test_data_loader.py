"""
Unit tests for src/data_loader.py.

Covers:
- Loading valid CSV files preserves row and column count
- Loading valid Excel files preserves row and column count
- Missing file raises FileNotFoundError
- Unsupported extension raises ValueError
- get_dataset_info returns correct shape and column names
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from src.data_loader import get_dataset_info, load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "name": ["Alice", "Bob", "Carol"],
    "gpa": [3.8, 3.2, 3.5],
    "career": ["Data Scientist", "Software Engineer", "Data Analyst"],
}


def _make_csv(tmp_path: str, data: dict | None = None) -> str:
    """Write a small CSV to *tmp_path* and return the file path."""
    df = pd.DataFrame(data or SAMPLE_DATA)
    path = os.path.join(tmp_path, "dataset.csv")
    df.to_csv(path, index=False)
    return path


def _make_excel(tmp_path: str, data: dict | None = None) -> str:
    """Write a small Excel file to *tmp_path* and return the file path."""
    df = pd.DataFrame(data or SAMPLE_DATA)
    path = os.path.join(tmp_path, "dataset.xlsx")
    df.to_excel(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_dataset — CSV
# ---------------------------------------------------------------------------


def test_load_csv_returns_dataframe(tmp_path):
    path = _make_csv(str(tmp_path))
    df = load_dataset(path)
    assert isinstance(df, pd.DataFrame)


def test_load_csv_preserves_row_count(tmp_path):
    path = _make_csv(str(tmp_path))
    df = load_dataset(path)
    assert len(df) == len(SAMPLE_DATA["name"])


def test_load_csv_preserves_column_count(tmp_path):
    path = _make_csv(str(tmp_path))
    df = load_dataset(path)
    assert len(df.columns) == len(SAMPLE_DATA)


def test_load_csv_preserves_column_names(tmp_path):
    path = _make_csv(str(tmp_path))
    df = load_dataset(path)
    assert list(df.columns) == list(SAMPLE_DATA.keys())


# ---------------------------------------------------------------------------
# load_dataset — Excel (.xlsx)
# ---------------------------------------------------------------------------


def test_load_xlsx_returns_dataframe(tmp_path):
    path = _make_excel(str(tmp_path))
    df = load_dataset(path)
    assert isinstance(df, pd.DataFrame)


def test_load_xlsx_preserves_row_count(tmp_path):
    path = _make_excel(str(tmp_path))
    df = load_dataset(path)
    assert len(df) == len(SAMPLE_DATA["name"])


def test_load_xlsx_preserves_column_count(tmp_path):
    path = _make_excel(str(tmp_path))
    df = load_dataset(path)
    assert len(df.columns) == len(SAMPLE_DATA)


def test_load_xlsx_preserves_column_names(tmp_path):
    path = _make_excel(str(tmp_path))
    df = load_dataset(path)
    assert list(df.columns) == list(SAMPLE_DATA.keys())


# ---------------------------------------------------------------------------
# load_dataset — Excel (.xls) via openpyxl/xlwt is not always available,
# so we test the extension routing by creating an xls-named file using xlrd-
# compatible content. We just verify the ValueError is NOT raised for .xls.
# ---------------------------------------------------------------------------


def test_load_xls_extension_is_accepted(tmp_path):
    """Verify .xls extension is routed to read_excel (not rejected as unsupported).

    Modern pandas no longer supports writing .xls (xlwt was removed), so we
    write an .xlsx file and copy it with a .xls suffix, then read it back via
    openpyxl which can open both extensions.
    """
    import shutil

    df_orig = pd.DataFrame(SAMPLE_DATA)
    xlsx_path = os.path.join(str(tmp_path), "dataset.xlsx")
    df_orig.to_excel(xlsx_path, index=False)
    # Rename to .xls — openpyxl can still read it under this extension
    xls_path = os.path.join(str(tmp_path), "dataset.xls")
    shutil.copy(xlsx_path, xls_path)
    df = load_dataset(xls_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(SAMPLE_DATA["name"])


# ---------------------------------------------------------------------------
# load_dataset — error cases
# ---------------------------------------------------------------------------


def test_missing_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path/to/dataset.csv")


def test_missing_file_error_message_contains_path():
    bad_path = "/nonexistent/dataset.csv"
    with pytest.raises(FileNotFoundError, match=bad_path):
        load_dataset(bad_path)


def test_unsupported_extension_raises_value_error(tmp_path):
    path = os.path.join(str(tmp_path), "dataset.json")
    pd.DataFrame(SAMPLE_DATA).to_json(path)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_unsupported_extension_error_message_contains_extension(tmp_path):
    path = os.path.join(str(tmp_path), "dataset.tsv")
    with open(path, "w") as f:
        f.write("col1\tcol2\n1\t2\n")
    with pytest.raises(ValueError, match=r"\.tsv"):
        load_dataset(path)


def test_unsupported_extension_txt_raises_value_error(tmp_path):
    path = os.path.join(str(tmp_path), "dataset.txt")
    with open(path, "w") as f:
        f.write("col1,col2\n1,2\n")
    with pytest.raises(ValueError):
        load_dataset(path)


# ---------------------------------------------------------------------------
# get_dataset_info
# ---------------------------------------------------------------------------


def test_get_dataset_info_returns_dict():
    df = pd.DataFrame(SAMPLE_DATA)
    info = get_dataset_info(df)
    assert isinstance(info, dict)


def test_get_dataset_info_row_count():
    df = pd.DataFrame(SAMPLE_DATA)
    info = get_dataset_info(df)
    assert info["row_count"] == 3


def test_get_dataset_info_column_count():
    df = pd.DataFrame(SAMPLE_DATA)
    info = get_dataset_info(df)
    assert info["column_count"] == 3


def test_get_dataset_info_column_names():
    df = pd.DataFrame(SAMPLE_DATA)
    info = get_dataset_info(df)
    assert info["columns"] == ["name", "gpa", "career"]


def test_get_dataset_info_empty_dataframe():
    df = pd.DataFrame()
    info = get_dataset_info(df)
    assert info["row_count"] == 0
    assert info["column_count"] == 0
    assert info["columns"] == []


def test_get_dataset_info_single_row():
    df = pd.DataFrame({"a": [1], "b": [2]})
    info = get_dataset_info(df)
    assert info["row_count"] == 1
    assert info["column_count"] == 2
