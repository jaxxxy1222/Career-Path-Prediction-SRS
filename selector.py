"""
Model selection and artifact serialization for the Career Path Prediction & Guidance System.

This module provides utilities to identify the best-performing trained model
from an evaluation summary and to persist the selected model, preprocessor,
and label encoder to disk so the web application can load them at startup
without retraining.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import pandas as pd

from src.preprocessor import Preprocessor


def select_best_model(
    evaluation_df: pd.DataFrame,
    models: dict[str, Any],
) -> tuple[str, Any]:
    """Return the model with the highest weighted F1-score.

    Identifies the row in *evaluation_df* with the maximum ``f1_score``
    value, then looks up the corresponding fitted model object from *models*
    using the ``model_name`` column of that row.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        Summary DataFrame produced by :meth:`~src.trainer.Trainer.evaluate_all`.
        Must contain at least the columns ``model_name`` and ``f1_score``.
        The integer index of each row corresponds to its position in the
        DataFrame (as returned by ``idxmax``).
    models : dict[str, Any]
        Mapping of model name (str) to a fitted estimator or Keras model.
        Keys must match the values in the ``model_name`` column of
        *evaluation_df*.

    Returns
    -------
    tuple[str, Any]
        A ``(model_name, model)`` pair where ``model_name`` is the string
        identifier of the best model and ``model`` is the corresponding
        fitted object from *models*.

    Examples
    --------
    >>> best_name, best_model = select_best_model(evaluation_df, models)
    >>> print(f"Best model: {best_name}")
    """
    best_idx = evaluation_df["f1_score"].idxmax()
    best_name = evaluation_df.loc[best_idx, "model_name"]
    best_model = models[best_name]
    return best_name, best_model


def save_artifacts(
    model: Any,
    preprocessor: Preprocessor,
    model_path: str,
    preprocessor_path: str,
    label_encoder: Optional[Any] = None,
    label_encoder_path: str = "artifacts/label_encoder.joblib",
) -> None:
    """Serialize model, preprocessor, and optionally a label encoder to disk.

    Uses :func:`joblib.dump` to persist each artifact.  The parent directory
    for each target path is created automatically if it does not already exist.

    Parameters
    ----------
    model : Any
        The fitted model (scikit-learn estimator or Keras model) to serialize.
    preprocessor : Preprocessor
        The fitted :class:`~src.preprocessor.Preprocessor` instance to
        serialize.
    model_path : str
        Destination file path for the model artifact
        (e.g. ``"artifacts/model.joblib"``).
    preprocessor_path : str
        Destination file path for the preprocessor artifact
        (e.g. ``"artifacts/preprocessor.joblib"``).
    label_encoder : optional
        A fitted :class:`~sklearn.preprocessing.LabelEncoder` (or any
        encoder that maps integer indices to career-label strings).  When
        provided it is serialized to *label_encoder_path*.
    label_encoder_path : str, optional
        Destination file path for the label encoder artifact.
        Defaults to ``"artifacts/label_encoder.joblib"``.

    Examples
    --------
    >>> save_artifacts(
    ...     model=best_model,
    ...     preprocessor=preprocessor,
    ...     model_path="artifacts/model.joblib",
    ...     preprocessor_path="artifacts/preprocessor.joblib",
    ...     label_encoder=le,
    ... )
    """
    # Ensure destination directories exist before writing.
    for path in [model_path, preprocessor_path]:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    if label_encoder is not None:
        le_parent = os.path.dirname(label_encoder_path)
        if le_parent:
            os.makedirs(le_parent, exist_ok=True)
        joblib.dump(label_encoder, label_encoder_path)
