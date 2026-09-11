"""
Preprocessing pipeline for the Career Path Prediction & Guidance System.

This module provides the :class:`Preprocessor` class, which fits and applies
imputation, encoding, and scaling transformations to student feature data, as
well as the :func:`stratified_split` utility for partitioning datasets while
preserving class distribution.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


class Preprocessor:
    """Fit and apply a full preprocessing pipeline to student feature data.

    The pipeline handles:

    - Median imputation for numeric columns
    - Mode (most-frequent) imputation for categorical columns
    - :class:`~sklearn.preprocessing.MinMaxScaler` normalization for numeric
      features (output range [0, 1])
    - :class:`~sklearn.preprocessing.OneHotEncoder` for nominal categorical
      features

    All transformation parameters are fitted **exclusively on training data**
    and subsequently applied unchanged to any other data (e.g., the test set or
    live inference inputs).

    A :exc:`UserWarning` is emitted during :meth:`fit` if any target class has
    fewer than 10 samples, indicating potential under-representation.

    Examples
    --------
    >>> pre = Preprocessor()
    >>> X_train_arr = pre.fit(X_train, y_train).transform(X_train)
    >>> X_test_arr  = pre.transform(X_test)
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Preprocessor":
        """Fit imputers, encoders, and scaler exclusively on training data.

        Identifies numeric and categorical columns automatically from *X*,
        builds a :class:`~sklearn.pipeline.Pipeline` for each column type,
        and fits the combined :class:`~sklearn.compose.ColumnTransformer`.

        A :exc:`UserWarning` is emitted for every target class in *y* that
        has fewer than 10 samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (training split only).
        y : pd.Series
            Target labels corresponding to *X*.

        Returns
        -------
        Preprocessor
            The fitted instance (enables method chaining).
        """
        self._check_class_sizes(y)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []

        if numeric_cols:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ])
            transformers.append(("numeric", numeric_pipeline, numeric_cols))

        if categorical_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        self._pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        self._pipeline.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transformations to *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to transform.  Columns must match those seen
            during :meth:`fit`.

        Returns
        -------
        np.ndarray
            Transformed feature array of shape ``(n_samples, n_features_out)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called before :meth:`transform`.
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError(
                "Preprocessor is not fitted. Call fit() or fit_transform() first."
            )
        result = self._pipeline.transform(X)
        # ColumnTransformer may return a sparse matrix; always return dense ndarray
        if hasattr(result, "toarray"):
            return result.toarray()
        return np.asarray(result)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fit the pipeline on *X* and immediately transform it.

        Convenience wrapper that is equivalent to calling
        ``preprocessor.fit(X, y).transform(X)``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (training split only).
        y : pd.Series
            Target labels corresponding to *X*.

        Returns
        -------
        np.ndarray
            Transformed feature array of shape ``(n_samples, n_features_out)``.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Serialize the fitted Preprocessor to disk using Joblib.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"artifacts/preprocessor.joblib"``).

        Raises
        ------
        RuntimeError
            If called before the preprocessor has been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted Preprocessor. Call fit() first."
            )
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Deserialize a :class:`Preprocessor` from disk.

        Parameters
        ----------
        path : str
            Path to the ``.joblib`` file previously created by :meth:`save`.

        Returns
        -------
        Preprocessor
            A fully fitted :class:`Preprocessor` ready to call
            :meth:`transform` on.

        Raises
        ------
        FileNotFoundError
            If no file exists at *path*.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Preprocessor artifact not found at '{path}'. "
                "Run the training pipeline first."
            )
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_class_sizes(y: pd.Series) -> None:
        """Emit a UserWarning for any class with fewer than 10 samples.

        Parameters
        ----------
        y : pd.Series
            Target labels for the training set.
        """
        counts = y.value_counts()
        for label, count in counts.items():
            if count < 10:
                warnings.warn(
                    f"Class '{label}' has only {count} sample(s) and may be "
                    "underrepresented. Consider collecting more data for this class.",
                    UserWarning,
                    stacklevel=4,
                )


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split *df* into stratified train and test sets.

    Uses :func:`~sklearn.model_selection.train_test_split` with
    ``stratify=y`` to preserve the class distribution of *target_col*
    across both partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including feature columns and the target column.
    target_col : str
        Name of the column in *df* that contains the class labels.
    test_size : float, optional
        Fraction of the dataset to include in the test split.
        Defaults to ``0.2`` (80/20 split).

    Returns
    -------
    X_train : pd.DataFrame
        Training feature matrix (80 % of rows by default).
    X_test : pd.DataFrame
        Test feature matrix (20 % of rows by default).
    y_train : pd.Series
        Training target labels.
    y_test : pd.Series
        Test target labels.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test
