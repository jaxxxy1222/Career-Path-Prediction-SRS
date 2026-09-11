"""
Unit tests for src/validator.py — validate_student_input.

Covers:
  - Missing required fields
  - Non-numeric GPA
  - GPA out of range
  - Empty / blank technical_skills and interests
  - Valid input
"""

import pytest
from src.validator import validate_student_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_input(**overrides) -> dict:
    """Return a baseline valid input dict, applying any overrides."""
    base = {
        "gpa": 3.5,
        "technical_skills": ["Python", "SQL"],
        "interests": ["Machine Learning"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

class TestMissingFields:
    def test_missing_gpa_returns_required_error(self):
        result = validate_student_input({"technical_skills": ["Python"], "interests": ["AI"]})
        valid, errors = result
        assert valid is False
        assert "gpa" in errors
        assert errors["gpa"] == "This field is required."

    def test_missing_technical_skills_returns_required_error(self):
        valid, errors = validate_student_input({"gpa": 3.0, "interests": ["AI"]})
        assert valid is False
        assert "technical_skills" in errors
        assert errors["technical_skills"] == "This field is required."

    def test_missing_interests_returns_required_error(self):
        valid, errors = validate_student_input({"gpa": 3.0, "technical_skills": ["Python"]})
        assert valid is False
        assert "interests" in errors
        assert errors["interests"] == "This field is required."

    def test_empty_dict_reports_all_three_fields(self):
        valid, errors = validate_student_input({})
        assert valid is False
        assert set(errors.keys()) == {"gpa", "technical_skills", "interests"}

    def test_none_gpa_returns_required_error(self):
        valid, errors = validate_student_input(_valid_input(gpa=None))
        assert valid is False
        assert "gpa" in errors

    def test_empty_string_gpa_returns_required_error(self):
        valid, errors = validate_student_input(_valid_input(gpa=""))
        assert valid is False
        assert "gpa" in errors


# ---------------------------------------------------------------------------
# GPA type validation (Requirements 8.4, 11.2)
# ---------------------------------------------------------------------------

class TestGpaTypeValidation:
    def test_non_numeric_string_returns_type_error(self):
        valid, errors = validate_student_input(_valid_input(gpa="abc"))
        assert valid is False
        assert errors["gpa"] == "GPA must be a numeric value."

    def test_alphabetic_with_digits_returns_type_error(self):
        valid, errors = validate_student_input(_valid_input(gpa="3.5x"))
        assert valid is False
        assert errors["gpa"] == "GPA must be a numeric value."

    def test_numeric_string_is_accepted(self):
        """Numeric strings like '85' should be coerced and accepted."""
        valid, errors = validate_student_input(_valid_input(gpa="85"))
        assert valid is True
        assert errors == {}

    def test_float_value_is_accepted(self):
        valid, errors = validate_student_input(_valid_input(gpa=3.5))
        assert valid is True

    def test_integer_value_is_accepted(self):
        valid, errors = validate_student_input(_valid_input(gpa=90))
        assert valid is True


# ---------------------------------------------------------------------------
# GPA range validation (Requirements 8.4, 11.2)
# ---------------------------------------------------------------------------

class TestGpaRangeValidation:
    def test_negative_gpa_returns_range_error(self):
        valid, errors = validate_student_input(_valid_input(gpa=-1))
        assert valid is False
        assert errors["gpa"] == "GPA must be between 0.0 and 100.0."

    def test_gpa_above_100_returns_range_error(self):
        valid, errors = validate_student_input(_valid_input(gpa=101))
        assert valid is False
        assert errors["gpa"] == "GPA must be between 0.0 and 100.0."

    def test_gpa_exactly_0_is_valid(self):
        valid, errors = validate_student_input(_valid_input(gpa=0.0))
        assert valid is True

    def test_gpa_exactly_100_is_valid(self):
        valid, errors = validate_student_input(_valid_input(gpa=100.0))
        assert valid is True

    def test_gpa_within_range_is_valid(self):
        valid, errors = validate_student_input(_valid_input(gpa=75.5))
        assert valid is True


# ---------------------------------------------------------------------------
# technical_skills validation (Requirements 8.3, 11.1)
# ---------------------------------------------------------------------------

class TestTechnicalSkillsValidation:
    def test_empty_list_returns_empty_error(self):
        valid, errors = validate_student_input(_valid_input(technical_skills=[]))
        assert valid is False
        assert errors["technical_skills"] == "Technical skills cannot be empty."

    def test_blank_string_returns_empty_error(self):
        valid, errors = validate_student_input(_valid_input(technical_skills="   "))
        assert valid is False
        assert errors["technical_skills"] == "Technical skills cannot be empty."

    def test_empty_string_returns_empty_error(self):
        valid, errors = validate_student_input(_valid_input(technical_skills=""))
        assert valid is False
        assert errors["technical_skills"] == "Technical skills cannot be empty."

    def test_non_empty_list_is_accepted(self):
        valid, errors = validate_student_input(_valid_input(technical_skills=["Python"]))
        assert valid is True

    def test_non_blank_string_is_accepted(self):
        valid, errors = validate_student_input(_valid_input(technical_skills="Python, SQL"))
        assert valid is True


# ---------------------------------------------------------------------------
# interests validation (Requirements 8.3, 11.1)
# ---------------------------------------------------------------------------

class TestInterestsValidation:
    def test_empty_list_returns_empty_error(self):
        valid, errors = validate_student_input(_valid_input(interests=[]))
        assert valid is False
        assert errors["interests"] == "Interests cannot be empty."

    def test_blank_string_returns_empty_error(self):
        valid, errors = validate_student_input(_valid_input(interests="\t\n"))
        assert valid is False
        assert errors["interests"] == "Interests cannot be empty."

    def test_empty_string_returns_empty_error(self):
        valid, errors = validate_student_input(_valid_input(interests=""))
        assert valid is False
        assert errors["interests"] == "Interests cannot be empty."

    def test_non_empty_list_is_accepted(self):
        valid, errors = validate_student_input(_valid_input(interests=["Data Science"]))
        assert valid is True

    def test_non_blank_string_is_accepted(self):
        valid, errors = validate_student_input(_valid_input(interests="AI, ML"))
        assert valid is True


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------

class TestValidInput:
    def test_fully_valid_dict_returns_true_and_empty_errors(self):
        valid, errors = validate_student_input({
            "gpa": 3.5,
            "technical_skills": ["Python", "SQL"],
            "interests": ["Machine Learning"],
        })
        assert valid is True
        assert errors == {}

    def test_valid_with_extra_fields_is_accepted(self):
        """Extra fields beyond the required three must not trigger errors."""
        valid, errors = validate_student_input({
            "gpa": 85.0,
            "technical_skills": ["Java"],
            "interests": ["Web Development"],
            "soft_skills": ["Communication"],
            "extracurricular": ["Chess Club"],
        })
        assert valid is True
        assert errors == {}

    def test_multiple_errors_collected_together(self):
        """All violations must be reported in a single call."""
        valid, errors = validate_student_input({
            "gpa": -5,
            "technical_skills": [],
            "interests": "   ",
        })
        assert valid is False
        assert "gpa" in errors
        assert "technical_skills" in errors
        assert "interests" in errors
