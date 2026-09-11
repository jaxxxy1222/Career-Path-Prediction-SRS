"""
Data loading utilities for the Career Path Prediction & Guidance System.

This module provides functions to load student datasets from CSV or Excel files
and to retrieve summary information about a loaded DataFrame.
"""

from __future__ import annotations

import os

import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a student dataset from a CSV or Excel file into a Pandas DataFrame.

    Supported formats:
    - ``.csv``  — loaded via :func:`pandas.read_csv`
    - ``.xlsx`` / ``.xls`` — loaded via :func:`pandas.read_excel`

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the dataset file.

    Returns
    -------
    pd.DataFrame
        The dataset loaded into a DataFrame with all original rows and columns
        preserved.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not point to an existing file.
    ValueError
        If the file extension is not one of ``.csv``, ``.xlsx``, or ``.xls``.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: '{filepath}'. "
            "Please provide a valid path to an existing file."
        )

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            "Expected one of: .csv, .xlsx, .xls"
        )


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return summary information about a loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame previously loaded by :func:`load_dataset`.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"row_count"`` (*int*) — number of rows in the DataFrame.
        - ``"column_count"`` (*int*) — number of columns in the DataFrame.
        - ``"columns"`` (*list[str]*) — ordered list of column names.
    """
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }
