"""
Custom exceptions for the Career Path Prediction & Guidance System.
"""


class PredictorError(Exception):
    """Raised when the Predictor encounters an error during inference.

    This exception wraps unexpected failures in the prediction pipeline
    (e.g., unexpected input shape, model inference failure) so that the
    UI layer can catch it and display a user-friendly message without
    exposing internal stack traces.
    """
