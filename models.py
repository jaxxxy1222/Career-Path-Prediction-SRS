"""
Shared data models for the Career Path Prediction & Guidance System.

This module defines the core dataclasses used across data processing,
model evaluation, prediction, and recommendation components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StudentInput:
    """Represents the academic profile submitted by a student.

    Parameters
    ----------
    gpa : float
        Overall GPA on a 0.0–4.0 or 0–100 scale.
    subject_scores : dict[str, float]
        Mapping of subject name to numeric score.
    extracurricular : list[str]
        Names of extracurricular activities the student participates in.
    technical_skills : list[str]
        Names of technical skills the student possesses.
    soft_skills : list[str]
        Names of soft skills the student possesses.
    interests : list[str]
        Names of professional/academic interest areas.
    """

    gpa: float
    subject_scores: dict[str, float] = field(default_factory=dict)
    extracurricular: list[str] = field(default_factory=list)
    technical_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Stores performance metrics for a single trained model.

    Parameters
    ----------
    model_name : str
        Human-readable identifier for the model (e.g., "Random Forest").
    accuracy : float
        Overall classification accuracy on the test set.
    precision : float
        Weighted precision across all classes.
    recall : float
        Weighted recall across all classes.
    f1_score : float
        Weighted F1-score across all classes.
    confusion_matrix : np.ndarray
        Square confusion matrix of shape (num_classes, num_classes).
    """

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray


@dataclass
class Course:
    """Represents a single online course recommendation.

    Parameters
    ----------
    title : str
        Full title of the course.
    provider : str
        Name of the platform or institution offering the course.
    url : str
        Direct URL where the course can be accessed.
    """

    title: str
    provider: str
    url: str


@dataclass
class PredictionResult:
    """Encapsulates a career prediction with its confidence and course recommendations.

    Parameters
    ----------
    career_label : str
        Predicted career path label (e.g., "Data Scientist").
    confidence_score : float
        Probability of the predicted class in the range [0.0, 1.0].
    courses : list[Course]
        Curated list of courses relevant to the predicted career label.
    """

    career_label: str
    confidence_score: float
    courses: list[Course] = field(default_factory=list)
