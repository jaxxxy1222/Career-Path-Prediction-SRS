"""
Model training and evaluation for the Career Path Prediction & Guidance System.

This module provides the :class:`Trainer` class, which trains classical
machine learning models (Random Forest, SVM, KNN, Logistic Regression) and
a deep learning model (Keras Sequential network), and evaluates all trained
models on a held-out test set.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class Trainer:
    """Train and evaluate classical ML and deep learning models.

    This class encapsulates all training logic for the career-path prediction
    pipeline.  Classical models are trained with scikit-learn estimators; the
    deep learning model is built and trained with TensorFlow/Keras.

    All public methods are stateless — results are returned rather than stored
    as instance attributes, making the :class:`Trainer` safe to reuse across
    multiple training runs.

    Examples
    --------
    >>> trainer = Trainer()
    >>> classical_models = trainer.train_classical_models(X_train, y_train)
    >>> dl_model = trainer.train_deep_learning_model(X_train, y_train, num_classes=10)
    >>> all_models = {**classical_models, "Neural Network": dl_model}
    >>> summary_df = trainer.evaluate_all(all_models, X_test, y_test)
    """

    # ------------------------------------------------------------------
    # Classical model training
    # ------------------------------------------------------------------

    def train_classical_models(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> dict[str, Any]:
        """Train Random Forest, SVM, KNN, and Logistic Regression classifiers.

        Each model is fitted on ``(X_train, y_train)``.  If any individual
        model raises an exception during fitting, the error is logged and
        training continues with the remaining models — the returned dict will
        simply omit that model's entry.

        Parameters
        ----------
        X_train : np.ndarray
            Preprocessed training feature matrix of shape
            ``(n_samples, n_features)``.
        y_train : np.ndarray
            Training target labels of shape ``(n_samples,)``.

        Returns
        -------
        dict[str, Any]
            Mapping of model name to fitted estimator.  Keys are:
            ``"Random Forest"``, ``"SVM"``, ``"KNN"``,
            ``"Logistic Regression"``.  A key is absent if that model
            failed during training.
        """
        candidates: dict[str, Any] = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_jobs=-1),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, n_jobs=-1
            ),
        }

        trained: dict[str, Any] = {}

        for name, model in candidates.items():
            start = time.perf_counter()
            try:
                model.fit(X_train, y_train)
                elapsed = time.perf_counter() - start
                logger.info(
                    "Trained '%s' successfully in %.2f seconds.", name, elapsed
                )
                trained[name] = model
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - start
                logger.error(
                    "Training '%s' failed after %.2f seconds: %s",
                    name,
                    elapsed,
                    exc,
                    exc_info=True,
                )

        return trained

    # ------------------------------------------------------------------
    # Deep learning model training
    # ------------------------------------------------------------------

    def train_deep_learning_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_classes: int,
    ) -> "keras.Model":  # type: ignore[name-defined]
        """Build and train a Keras Sequential neural network.

        Architecture
        ------------
        - **Input**: shape ``(input_dim,)``
        - **Hidden layer 1**: Dense(256, ReLU) → BatchNormalization → Dropout(0.3)
        - **Hidden layer 2**: Dense(128, ReLU) → BatchNormalization → Dropout(0.3)
        - **Output**: Dense(``num_classes``, softmax)

        Training uses the **Adam** optimizer with
        ``sparse_categorical_crossentropy`` loss and early stopping
        (``patience=5``, ``monitor="val_loss"``,
        ``restore_best_weights=True``).

        Parameters
        ----------
        X_train : np.ndarray
            Preprocessed training feature matrix of shape
            ``(n_samples, n_features)``.
        y_train : np.ndarray
            Integer-encoded training target labels of shape ``(n_samples,)``.
        num_classes : int
            Number of unique career-label classes (determines output layer
            size).

        Returns
        -------
        keras.Model
            The fitted Keras model with the best weights restored by early
            stopping.
        """
        # Import here to avoid penalising users who only need classical models.
        import tensorflow as tf
        from tensorflow import keras

        input_dim = X_train.shape[1]

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(input_dim,)),
                # Hidden layer 1
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                # Hidden layer 2
                keras.layers.Dense(128, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                # Output layer
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        start = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )
        elapsed = time.perf_counter() - start
        logger.info("Trained 'Neural Network' in %.2f seconds.", elapsed)

        return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Compute classification metrics for a single trained model.

        Supports any model that exposes a ``predict`` method compatible with
        scikit-learn (returns class labels) *or* a Keras model (whose
        ``predict`` returns probability arrays that are argmax-reduced).

        Parameters
        ----------
        model : Any
            A fitted scikit-learn estimator or a compiled/trained Keras model.
        X_test : np.ndarray
            Preprocessed test feature matrix of shape
            ``(n_samples, n_features)``.
        y_test : np.ndarray
            True target labels of shape ``(n_samples,)``.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"accuracy"`` — overall accuracy (float)
            - ``"precision"`` — weighted precision (float)
            - ``"recall"`` — weighted recall (float)
            - ``"f1_score"`` — weighted F1-score (float)
            - ``"confusion_matrix"`` — :class:`numpy.ndarray` of shape
              ``(n_classes, n_classes)``
        """
        # Keras models return probability arrays; resolve to class indices.
        try:
            import tensorflow as tf
            from tensorflow import keras as _keras
            if isinstance(model, _keras.Model):
                proba = model.predict(X_test, verbose=0)
                y_pred = np.argmax(proba, axis=1)
            else:
                y_pred = model.predict(X_test)
        except ImportError:
            y_pred = model.predict(X_test)

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def evaluate_all(
        self,
        models: dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate every model in *models* on the same test set.

        Calls :meth:`evaluate_model` for each entry in *models* and
        assembles the scalar metrics (accuracy, precision, recall, F1) into
        a summary :class:`pandas.DataFrame`.  The confusion matrix is
        excluded from the DataFrame but is accessible by calling
        :meth:`evaluate_model` directly.

        Parameters
        ----------
        models : dict
            Mapping of model name (str) to a fitted estimator or Keras model.
        X_test : np.ndarray
            Preprocessed test feature matrix of shape
            ``(n_samples, n_features)``.
        y_test : np.ndarray
            True target labels of shape ``(n_samples,)``.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per model and columns:
            ``model_name``, ``accuracy``, ``precision``, ``recall``,
            ``f1_score``.
        """
        rows = []
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            rows.append(
                {
                    "model_name": name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                }
            )
        return pd.DataFrame(rows, columns=["model_name", "accuracy", "precision", "recall", "f1_score"])
