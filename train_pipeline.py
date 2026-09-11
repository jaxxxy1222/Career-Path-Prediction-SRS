"""
Offline training entry point for the Career Path Prediction & Guidance System.

Run this script to orchestrate the full training pipeline:
  1. Load dataset (CSV or Excel)
  2. Print dataset info
  3. Run exploratory data analysis (EDA)
  4. Stratified train/test split
  5. Fit and transform training set with Preprocessor
  6. Transform test set with the same fitted Preprocessor
  7. Train classical ML models (Random Forest, SVM, KNN, Logistic Regression)
  8. Train deep learning model (TensorFlow/Keras, optional)
  9. Evaluate all models on the test set
  10. Print evaluation summary table
  11. Select the best model by weighted F1-score
  12. Fit and save a LabelEncoder for the target column
  13. Save all artifacts (model, preprocessor, label encoder) to disk

Usage
-----
    python train_pipeline.py --data path/to/dataset.csv
    python train_pipeline.py --data path/to/dataset.xlsx --target CareerColumn

Requirements: 1.1, 1.2, 2.1-2.7, 3.1-3.5, 4.1-4.4, 5.1-5.5, 6.1-6.4, 7.1-7.3
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_dataset, get_dataset_info
from src.eda_module import run_eda
from src.preprocessor import Preprocessor, stratified_split
from src.trainer import Trainer
from src.selector import select_best_model, save_artifacts

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``--data`` (required) and ``--target`` (optional)
        arguments.
    """
    parser = argparse.ArgumentParser(
        prog="train_pipeline",
        description=(
            "Offline training pipeline for the Career Path Prediction system. "
            "Loads a dataset, runs EDA, trains ML/DL models, evaluates them, "
            "selects the best model, and saves all artifacts to disk."
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to the training dataset (CSV or Excel .xlsx/.xls).",
    )
    parser.add_argument(
        "--target",
        default=None,
        metavar="COLUMN",
        help=(
            "Name of the target column containing career labels. "
            "Defaults to the last column in the dataset."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _resolve_target_column(df: pd.DataFrame, target_arg: str | None) -> str:
    """Return the target column name, defaulting to the last column.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    target_arg : str or None
        Value of the ``--target`` CLI argument; ``None`` means use last column.

    Returns
    -------
    str
        Column name to treat as the classification target.

    Raises
    ------
    ValueError
        If ``target_arg`` is not None and is not a column in *df*.
    """
    if target_arg is not None:
        if target_arg not in df.columns:
            raise ValueError(
                f"Target column '{target_arg}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        return target_arg
    return df.columns[-1]


def _print_evaluation_table(evaluation_df: pd.DataFrame) -> None:
    """Print a formatted evaluation summary table to stdout.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        DataFrame produced by :meth:`~src.trainer.Trainer.evaluate_all`.
        Expected columns: ``model_name``, ``accuracy``, ``precision``,
        ``recall``, ``f1_score``.
    """
    col_widths = {
        "model_name": 22,
        "accuracy": 10,
        "precision": 11,
        "recall": 10,
        "f1_score": 10,
    }
    header = (
        f"{'Model':<{col_widths['model_name']}}"
        f"{'Accuracy':>{col_widths['accuracy']}}"
        f"{'Precision':>{col_widths['precision']}}"
        f"{'Recall':>{col_widths['recall']}}"
        f"{'F1-Score':>{col_widths['f1_score']}}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)
    for _, row in evaluation_df.iterrows():
        print(
            f"{str(row['model_name']):<{col_widths['model_name']}}"
            f"{row['accuracy']:>{col_widths['accuracy']}.4f}"
            f"{row['precision']:>{col_widths['precision']}.4f}"
            f"{row['recall']:>{col_widths['recall']}.4f}"
            f"{row['f1_score']:>{col_widths['f1_score']}.4f}"
        )
    print(separator + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, target_col_arg: str | None = None) -> None:
    """Execute the full offline training pipeline.

    Orchestrates all pipeline stages in order:

    1. Load dataset from *data_path*.
    2. Print dataset info.
    3. Run EDA (saves PNGs to ``eda_output/``).
    4. Stratified train/test split (80/20).
    5. Fit + transform training features with :class:`~src.preprocessor.Preprocessor`.
    6. Transform test features with the same fitted Preprocessor.
    7. Train classical ML models.
    8. Train deep learning model (gracefully skipped if TensorFlow is absent).
    9. Evaluate all models on the held-out test set.
    10. Print evaluation summary table to stdout.
    11. Select the best model by weighted F1-score.
    12. Fit a :class:`~sklearn.preprocessing.LabelEncoder` on the target column.
    13. Save model, preprocessor, and label encoder to ``artifacts/``.

    Parameters
    ----------
    data_path : str
        Path to the CSV or Excel dataset file.
    target_col_arg : str or None
        Explicit target column name, or ``None`` to use the last column.

    Raises
    ------
    FileNotFoundError
        If *data_path* does not point to an existing file.
    ValueError
        If the file format is unsupported or the target column is not found.
    """
    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset from '%s' …", data_path)
    df = load_dataset(data_path)

    # ------------------------------------------------------------------
    # Step 2: Print dataset info
    # ------------------------------------------------------------------
    info = get_dataset_info(df)
    print(
        f"\nDataset loaded:  {info['row_count']} rows × "
        f"{info['column_count']} columns"
    )
    print(f"Columns:  {info['columns']}\n")

    # ------------------------------------------------------------------
    # Step 3: Run EDA
    # ------------------------------------------------------------------
    logger.info("Running EDA …")
    eda_paths = run_eda(df, output_dir="eda_output")
    logger.info("EDA complete — %d plot(s) saved to eda_output/", len(eda_paths))

    # ------------------------------------------------------------------
    # Step 4: Resolve target column and stratified split
    # ------------------------------------------------------------------
    target_col = _resolve_target_column(df, target_col_arg)
    logger.info("Target column: '%s'", target_col)

    X_train, X_test, y_train, y_test = stratified_split(df, target_col)
    logger.info(
        "Train/test split:  %d train samples, %d test samples",
        len(X_train),
        len(X_test),
    )

    # ------------------------------------------------------------------
    # Step 5 & 6: Fit Preprocessor on training set, transform both sets
    # ------------------------------------------------------------------
    logger.info("Fitting Preprocessor on training data …")
    preprocessor = Preprocessor()
    X_train_arr = preprocessor.fit_transform(X_train, y_train)
    logger.info(
        "Training features transformed: shape %s", X_train_arr.shape
    )

    X_test_arr = preprocessor.transform(X_test)
    logger.info("Test features transformed:     shape %s", X_test_arr.shape)

    # ------------------------------------------------------------------
    # Step 7: Train classical ML models
    # ------------------------------------------------------------------
    logger.info("Training classical ML models …")
    trainer = Trainer()
    classical_models = trainer.train_classical_models(X_train_arr, y_train.to_numpy())
    logger.info(
        "Classical models trained: %s", list(classical_models.keys())
    )

    # ------------------------------------------------------------------
    # Step 8: Train deep learning model (optional — needs TensorFlow)
    # ------------------------------------------------------------------
    all_models: dict = dict(classical_models)
    num_classes = df[target_col].nunique()

    try:
        logger.info(
            "Training deep learning model (%d classes, %d features) …",
            num_classes,
            X_train_arr.shape[1],
        )
        # Encode labels as integers for Keras sparse_categorical_crossentropy
        le_temp = LabelEncoder()
        y_train_enc = le_temp.fit_transform(y_train.to_numpy())

        dl_model = trainer.train_deep_learning_model(
            X_train_arr, y_train_enc, num_classes=num_classes
        )
        all_models["Neural Network"] = dl_model
        logger.info("Deep learning model trained successfully.")
    except ImportError:
        logger.warning(
            "TensorFlow is not installed — deep learning model skipped."
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Deep learning training failed and will be skipped: %s", exc,
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # Step 9: Evaluate all models
    # ------------------------------------------------------------------
    logger.info("Evaluating all models on the test set …")

    # For the DL model, y_test must also be integer-encoded
    le_for_eval = LabelEncoder().fit(y_train.to_numpy())
    y_test_int = le_for_eval.transform(y_test.to_numpy())

    # Build evaluation-compatible copies: DL model needs integer labels,
    # classical models work with raw string labels.  We pass the appropriate
    # y_test to evaluate_all by temporarily wrapping DL evaluation separately.
    classical_only = {k: v for k, v in all_models.items() if k != "Neural Network"}
    evaluation_rows = []

    if classical_only:
        classical_df = trainer.evaluate_all(
            classical_only, X_test_arr, y_test.to_numpy()
        )
        evaluation_rows.append(classical_df)

    if "Neural Network" in all_models:
        dl_df = trainer.evaluate_all(
            {"Neural Network": all_models["Neural Network"]},
            X_test_arr,
            y_test_int,
        )
        evaluation_rows.append(dl_df)

    if not evaluation_rows:
        logger.error("No models were successfully trained. Exiting.")
        sys.exit(1)

    evaluation_df = pd.concat(evaluation_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Step 10: Print evaluation summary table
    # ------------------------------------------------------------------
    print("\n=== Model Evaluation Summary ===")
    _print_evaluation_table(evaluation_df)

    # ------------------------------------------------------------------
    # Step 11: Select best model
    # ------------------------------------------------------------------
    best_name, best_model = select_best_model(evaluation_df, all_models)
    logger.info("Best model: '%s'", best_name)
    print(f"Best model selected:  {best_name}")
    best_row = evaluation_df[evaluation_df["model_name"] == best_name].iloc[0]
    print(
        f"  Accuracy:   {best_row['accuracy']:.4f}\n"
        f"  Precision:  {best_row['precision']:.4f}\n"
        f"  Recall:     {best_row['recall']:.4f}\n"
        f"  F1-Score:   {best_row['f1_score']:.4f}\n"
    )

    # ------------------------------------------------------------------
    # Step 12: Fit LabelEncoder on full target column
    # ------------------------------------------------------------------
    label_encoder = LabelEncoder()
    label_encoder.fit(df[target_col].astype(str))
    logger.info(
        "LabelEncoder fitted on %d unique classes.", len(label_encoder.classes_)
    )

    # ------------------------------------------------------------------
    # Step 13: Save artifacts
    # ------------------------------------------------------------------
    logger.info("Saving artifacts to artifacts/ …")
    save_artifacts(
        model=best_model,
        preprocessor=preprocessor,
        model_path="artifacts/model.joblib",
        preprocessor_path="artifacts/preprocessor.joblib",
        label_encoder=label_encoder,
        label_encoder_path="artifacts/label_encoder.joblib",
    )
    logger.info("Artifacts saved successfully.")
    print(
        "Artifacts saved:\n"
        "  artifacts/model.joblib\n"
        "  artifacts/preprocessor.joblib\n"
        "  artifacts/label_encoder.joblib\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and invoke :func:`run_pipeline`.

    Exits with code 1 on :exc:`FileNotFoundError` or :exc:`ValueError`,
    logging a descriptive message before exiting.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(data_path=args.data, target_col_arg=args.target)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Value error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
