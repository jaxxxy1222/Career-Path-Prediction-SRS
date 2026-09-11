"""
Inference module for the Career Path Prediction & Guidance System.

This module provides the :class:`Predictor` class, which loads serialized
model, preprocessor, and label-encoder artifacts from disk at initialization
and exposes a :meth:`~Predictor.predict` method for single-input career-path
inference.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from src.exceptions import PredictorError

# Default artifact paths (relative to the project root)
_DEFAULT_MODEL_PATH = "artifacts/model.joblib"
_DEFAULT_PREPROCESSOR_PATH = "artifacts/preprocessor.joblib"
_DEFAULT_LABEL_ENCODER_PATH = "artifacts/label_encoder.joblib"


class Predictor:
    """Load serialized artifacts and serve career-path predictions.

    On instantiation the class loads three artifacts from disk:

    - **model** – the best fitted scikit-learn estimator or Keras model,
      selected and serialized by the training pipeline.
    - **preprocessor** – the fitted :class:`~src.preprocessor.Preprocessor`
      instance that encodes and scales raw input features.
    - **label_encoder** – a fitted :class:`~sklearn.preprocessing.LabelEncoder`
      (or equivalent) that maps integer class indices back to human-readable
      career-label strings.

    Parameters
    ----------
    model_path : str, optional
        Path to the serialized model artifact.
        Defaults to ``"artifacts/model.joblib"``.
    preprocessor_path : str, optional
        Path to the serialized preprocessor artifact.
        Defaults to ``"artifacts/preprocessor.joblib"``.
    label_encoder_path : str, optional
        Path to the serialized label-encoder artifact.
        Defaults to ``"artifacts/label_encoder.joblib"``.

    Raises
    ------
    FileNotFoundError
        If any of the three artifact files is missing from disk, with the
        message ``"Run training pipeline first."``.

    Examples
    --------
    >>> predictor = Predictor()
    >>> career, confidence = predictor.predict({"gpa": 3.8, "major": "CS"})
    >>> print(career, confidence)
    Data Scientist 0.92
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        preprocessor_path: str = _DEFAULT_PREPROCESSOR_PATH,
        label_encoder_path: str = _DEFAULT_LABEL_ENCODER_PATH,
    ) -> None:
        # Validate all artifact paths before attempting to load anything so
        # the error message is consistent regardless of which file is absent.
        for path in (model_path, preprocessor_path, label_encoder_path):
            if not os.path.exists(path):
                raise FileNotFoundError("Run training pipeline first.")

        self._model = joblib.load(model_path)
        self._preprocessor = joblib.load(preprocessor_path)
        self._label_encoder = joblib.load(label_encoder_path)

    def predict(self, raw_input: dict) -> tuple[str, float]:
        """Apply the preprocessing pipeline and run inference on *raw_input*.

        Converts the input dictionary to a single-row
        :class:`~pandas.DataFrame`, applies the fitted preprocessor, runs the
        model to obtain class probabilities, and returns the most likely career
        label together with its probability.

        For scikit-learn estimators the method uses ``predict_proba``; for
        Keras models (which expose ``predict`` instead of ``predict_proba``)
        the raw softmax output is treated as the probability distribution.

        Parameters
        ----------
        raw_input : dict
            A mapping of feature name to feature value representing a single
            student profile.  Keys must match the column names seen by the
            preprocessor during training.

        Returns
        -------
        tuple[str, float]
            A ``(career_label, confidence_score)`` pair where *career_label*
            is the predicted career-path string and *confidence_score* is the
            corresponding class probability in the range [0.0, 1.0].

        Raises
        ------
        PredictorError
            Wraps any exception that occurs during preprocessing or inference,
            ensuring the UI layer never surfaces raw stack traces.

        Examples
        --------
        >>> career, score = predictor.predict({"gpa": 3.5, "major": "ECE"})
        >>> 0.0 <= score <= 1.0
        True
        """
        try:
            # Convert raw dict to a single-row DataFrame so the preprocessor
            # receives the column names it was fitted on.
            input_df = pd.DataFrame([raw_input])

            # Apply the fitted preprocessing pipeline.
            X_transformed = self._preprocessor.transform(input_df)

            # Obtain class probabilities.  Scikit-learn estimators expose
            # predict_proba; Keras models expose predict and return softmax
            # probabilities directly.
            if hasattr(self._model, "predict_proba"):
                probabilities = self._model.predict_proba(X_transformed)
            else:
                # Keras / TensorFlow model
                probabilities = self._model.predict(X_transformed)

            # probabilities shape: (1, num_classes)
            proba_row = probabilities[0]
            best_idx = int(np.argmax(proba_row))
            max_prob = float(proba_row[best_idx])

            # Decode the integer index back to a career-label string.
            career_label: str = self._label_encoder.inverse_transform([best_idx])[0]

            return career_label, max_prob

        except Exception as exc:
            raise PredictorError(
                f"Prediction failed: {exc}"
            ) from exc
