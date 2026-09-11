"""
Exploratory Data Analysis (EDA) module for the Career Path Prediction & Guidance System.

This module provides functions to generate and save visualizations of the dataset,
including distribution plots for numeric features, a correlation heatmap, and a
class balance bar chart.  All plots are saved as ``.png`` files to a configurable
output directory that is created automatically if it does not exist.

Usage example::

    import pandas as pd
    from src.eda_module import run_eda

    df = pd.read_csv("data/students.csv")
    saved_paths = run_eda(df, output_dir="eda_output")
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate all EDA plots, save them to *output_dir*, and return saved paths.

    Orchestrates the full EDA suite by calling
    :func:`plot_distributions`, :func:`plot_correlation_heatmap`, and
    :func:`plot_class_balance`.  The output directory is created if it does not
    already exist.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.  Must contain at least one numeric column and a
        ``"Career_Label"`` column (or the last column is used as the class label
        when ``"Career_Label"`` is absent).
    output_dir : str
        Directory path where ``.png`` files will be written.

    Returns
    -------
    list[str]
        Sorted list of absolute or relative file paths for every PNG that was
        saved during the EDA run.
    """
    os.makedirs(output_dir, exist_ok=True)

    saved: list[str] = []

    # 1. Feature distributions (one PNG per numeric column)
    saved.extend(plot_distributions(df, output_dir))

    # 2. Correlation heatmap (one PNG)
    saved.append(plot_correlation_heatmap(df, output_dir))

    # 3. Class balance (one PNG)
    target_col = _infer_target_column(df)
    saved.append(plot_class_balance(df[target_col], output_dir))

    return saved


def plot_distributions(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate a histogram + KDE distribution plot for each numeric feature.

    One ``.png`` file is produced per numeric column (columns whose dtype is a
    numeric type as reported by :func:`~pandas.DataFrame.select_dtypes`).
    Non-numeric columns are silently skipped.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset whose numeric columns should be visualised.
    output_dir : str
        Directory where the PNG files are saved.  Created automatically if it
        does not exist.

    Returns
    -------
    list[str]
        File paths of the saved distribution plots, one per numeric column.
        Returns an empty list when *df* contains no numeric columns.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    saved: list[str] = []

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}", fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        plt.tight_layout()

        filename = _safe_filename(col)
        filepath = os.path.join(output_dir, f"dist_{filename}.png")
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved.append(filepath)

    return saved


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Generate a correlation heatmap of all numeric features and save it.

    Only numeric columns are included in the correlation matrix.  If the
    DataFrame contains no numeric columns, an empty heatmap is saved.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to compute pairwise Pearson correlations for.
    output_dir : str
        Directory where the PNG file is saved.  Created automatically if it
        does not exist.

    Returns
    -------
    str
        File path of the saved heatmap PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()

    # Size scales with number of features (minimum 8×6)
    n_cols = max(len(numeric_df.columns), 1)
    fig_size = max(8, n_cols * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.75))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return filepath


def plot_class_balance(y: pd.Series, output_dir: str) -> str:
    """Generate a bar chart showing the sample count per career label.

    The bars are sorted in descending order of count to make imbalances easy
    to spot.  The count is annotated above each bar.

    Parameters
    ----------
    y : pd.Series
        Target label column (``Career_Label`` or equivalent) from the dataset.
    output_dir : str
        Directory where the PNG file is saved.  Created automatically if it
        does not exist.

    Returns
    -------
    str
        File path of the saved class balance PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    counts = y.value_counts().sort_values(ascending=False)

    n_classes = max(len(counts), 1)
    fig_width = max(10, n_classes * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars = ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white")

    # Annotate count above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Class Balance (Samples per Career Label)", fontsize=14)
    ax.set_xlabel("Career Label", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "class_balance.png")
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return filepath


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _infer_target_column(df: pd.DataFrame) -> str:
    """Return the name of the target (class label) column in *df*.

    Checks for a column named ``"Career_Label"`` (case-insensitive).  If not
    found, falls back to the last column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset including the target column.

    Returns
    -------
    str
        Column name to use as the class label.
    """
    for col in df.columns:
        if col.strip().lower() == "career_label":
            return col
    # Fallback: treat the last column as the target
    return df.columns[-1]


def _safe_filename(column_name: str) -> str:
    """Convert a column name to a filesystem-safe string.

    Replaces spaces and special characters with underscores and strips
    leading/trailing whitespace so the result is usable as a file name
    component.

    Parameters
    ----------
    column_name : str
        The raw column name to sanitise.

    Returns
    -------
    str
        A sanitised string suitable for use in a file path.
    """
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in column_name.strip())
    # Collapse consecutive underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "column"
