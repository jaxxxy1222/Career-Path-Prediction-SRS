"""
Unit tests for src/trainer.py — Trainer class.

Covers:
- train_classical_models: returns 4 models, handles individual failures gracefully
- train_deep_learning_model: correct architecture and output layer
- evaluate_model: correct metrics for known labels
- evaluate_all: returns a DataFrame with expected shape and columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.trainer import Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_dataset():
    """Small, linearly-separable dataset with 3 classes."""
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    # 80/20 split
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture()
def trainer():
    return Trainer()


# ---------------------------------------------------------------------------
# train_classical_models
# ---------------------------------------------------------------------------

class TestTrainClassicalModels:
    def test_returns_four_models(self, trainer, simple_dataset):
        """train_classical_models returns exactly four fitted models."""
        X_train, _, y_train, _ = simple_dataset
        models = trainer.train_classical_models(X_train, y_train)
        assert len(models) == 4
        assert set(models.keys()) == {
            "Random Forest", "SVM", "KNN", "Logistic Regression"
        }

    def test_all_models_can_predict(self, trainer, simple_dataset):
        """Every returned model can call predict without error."""
        X_train, X_test, y_train, _ = simple_dataset
        models = trainer.train_classical_models(X_train, y_train)
        for name, model in models.items():
            preds = model.predict(X_test)
            assert len(preds) == len(X_test), f"{name} returned wrong prediction count"

    def test_model_failure_does_not_stop_others(self, trainer, monkeypatch):
        """An exception in one model does not prevent the rest from training."""
        # Patch RandomForestClassifier.fit to raise
        from sklearn.ensemble import RandomForestClassifier

        original_fit = RandomForestClassifier.fit

        def bad_fit(self, X, y, **kwargs):
            raise RuntimeError("Simulated training failure")

        monkeypatch.setattr(RandomForestClassifier, "fit", bad_fit)

        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2,
            n_clusters_per_class=1, random_state=0
        )
        models = trainer.train_classical_models(X, y)

        # Random Forest should be absent; the other 3 should be present
        assert "Random Forest" not in models
        assert len(models) == 3
        for name in ("SVM", "KNN", "Logistic Regression"):
            assert name in models


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_perfect_predictions_give_accuracy_one(self, trainer):
        """A model that predicts all labels correctly gets accuracy=1.0."""
        X = np.zeros((20, 3))
        y = np.array([0, 1, 2] * 6 + [0, 2])
        dummy = DummyClassifier(strategy="constant", constant=0)
        dummy.fit(X, y)

        # Build a model that mirrors the true labels perfectly
        perfect = LogisticRegression(max_iter=1000, random_state=42)
        # Just need it fitted — we'll override predict
        perfect.fit(X, y)

        class PerfectModel:
            def predict(self, X):
                return y

        metrics = trainer.evaluate_model(PerfectModel(), X, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_score"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_metrics_dict_has_required_keys(self, trainer, simple_dataset):
        """evaluate_model output contains all required metric keys."""
        X_train, X_test, y_train, y_test = simple_dataset
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        metrics = trainer.evaluate_model(model, X_test, y_test)
        for key in ("accuracy", "precision", "recall", "f1_score", "confusion_matrix"):
            assert key in metrics, f"Missing key: {key}"

    def test_confusion_matrix_shape(self, trainer, simple_dataset):
        """Confusion matrix shape is (num_classes, num_classes)."""
        X_train, X_test, y_train, y_test = simple_dataset
        num_classes = len(np.unique(y_train))
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        metrics = trainer.evaluate_model(model, X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert cm.shape == (num_classes, num_classes)

    def test_metrics_are_in_valid_range(self, trainer, simple_dataset):
        """All scalar metrics are floats in [0.0, 1.0]."""
        X_train, X_test, y_train, y_test = simple_dataset
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        metrics = trainer.evaluate_model(model, X_test, y_test)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of [0,1]: {metrics[key]}"


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def test_returns_dataframe_with_correct_columns(self, trainer, simple_dataset):
        """evaluate_all returns a DataFrame with the expected columns."""
        X_train, X_test, y_train, y_test = simple_dataset
        models = trainer.train_classical_models(X_train, y_train)
        df = trainer.evaluate_all(models, X_test, y_test)

        assert isinstance(df, pd.DataFrame)
        for col in ("model_name", "accuracy", "precision", "recall", "f1_score"):
            assert col in df.columns, f"Column '{col}' missing from summary DataFrame"

    def test_returns_one_row_per_model(self, trainer, simple_dataset):
        """evaluate_all produces exactly one row per model passed in."""
        X_train, X_test, y_train, y_test = simple_dataset
        models = trainer.train_classical_models(X_train, y_train)
        df = trainer.evaluate_all(models, X_test, y_test)
        assert len(df) == len(models)

    def test_model_names_match(self, trainer, simple_dataset):
        """Model names in the summary DataFrame match the input dict keys."""
        X_train, X_test, y_train, y_test = simple_dataset
        models = trainer.train_classical_models(X_train, y_train)
        df = trainer.evaluate_all(models, X_test, y_test)
        assert set(df["model_name"].tolist()) == set(models.keys())


# ---------------------------------------------------------------------------
# train_deep_learning_model (architecture checks)
# ---------------------------------------------------------------------------

class TestTrainDeepLearningModel:
    """Architecture and output shape tests for the Keras model."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self):
        pytest.importorskip("tensorflow", reason="TensorFlow not installed")

    def test_model_output_units_equals_num_classes(self, trainer, simple_dataset):
        """Output layer has exactly num_classes units."""
        X_train, _, y_train, _ = simple_dataset
        num_classes = len(np.unique(y_train))
        model = trainer.train_deep_learning_model(X_train, y_train, num_classes)
        output_units = model.layers[-1].units
        assert output_units == num_classes

    def test_output_layer_activation_is_softmax(self, trainer, simple_dataset):
        """Output layer uses softmax activation."""
        import tensorflow as tf
        X_train, _, y_train, _ = simple_dataset
        num_classes = len(np.unique(y_train))
        model = trainer.train_deep_learning_model(X_train, y_train, num_classes)
        activation = model.layers[-1].activation
        assert activation == tf.keras.activations.softmax

    def test_model_has_at_least_two_dense_hidden_layers(self, trainer, simple_dataset):
        """The network has at least 2 Dense hidden layers (excluding output)."""
        from tensorflow.keras.layers import Dense
        X_train, _, y_train, _ = simple_dataset
        num_classes = len(np.unique(y_train))
        model = trainer.train_deep_learning_model(X_train, y_train, num_classes)

        # All Dense layers excluding the output layer
        dense_hidden = [
            layer for layer in model.layers[:-1]
            if isinstance(layer, Dense)
        ]
        assert len(dense_hidden) >= 2

    def test_input_shape_matches_feature_count(self, trainer, simple_dataset):
        """The model's input shape matches the number of training features."""
        X_train, _, y_train, _ = simple_dataset
        num_classes = len(np.unique(y_train))
        model = trainer.train_deep_learning_model(X_train, y_train, num_classes)
        # model.input_shape is (None, n_features)
        assert model.input_shape == (None, X_train.shape[1])

    def test_model_can_predict_on_test_set(self, trainer, simple_dataset):
        """Trained Keras model produces probability arrays of correct shape."""
        X_train, X_test, y_train, _ = simple_dataset
        num_classes = len(np.unique(y_train))
        model = trainer.train_deep_learning_model(X_train, y_train, num_classes)

        proba = model.predict(X_test, verbose=0)
        assert proba.shape == (len(X_test), num_classes)
        # Each row should sum to ~1 (softmax output)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-5)
