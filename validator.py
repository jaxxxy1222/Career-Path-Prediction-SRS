"""
Input validation for the Career Path Prediction & Guidance System.

This module validates raw student form submissions before they are passed
to the prediction pipeline, ensuring all required fields are present and
contain values that are semantically correct.
"""

from __future__ import annotations


def validate_student_input(raw_input: dict) -> tuple[bool, dict[str, str]]:
    """Validate a raw student form submission.

    Checks that all required fields are present and that each value
    satisfies the domain constraints described below.

    Required fields
    ---------------
    - ``gpa`` : must be a numeric value in the range [0.0, 100.0].
    - ``technical_skills`` : must be a non-empty list or a non-blank string.
    - ``interests`` : must be a non-empty list or a non-blank string.

    Parameters
    ----------
    raw_input : dict
        Mapping of field names to their submitted values.

    Returns
    -------
    tuple[bool, dict[str, str]]
        A two-element tuple:

        - ``is_valid`` (bool): ``True`` when all fields pass validation,
          ``False`` if one or more violations are detected.
        - ``errors`` (dict[str, str]): Maps each failing field name to a
          human-readable error message.  Empty when ``is_valid`` is ``True``.

    Examples
    --------
    >>> validate_student_input({"gpa": 3.5, "technical_skills": ["Python"], "interests": ["AI"]})
    (True, {})
    >>> validate_student_input({"gpa": "abc", "technical_skills": [], "interests": ["AI"]})
    (False, {'gpa': 'GPA must be a numeric value.', 'technical_skills': 'Technical skills cannot be empty.'})
    """
    errors: dict[str, str] = {}

    # --- gpa ---
    if "gpa" not in raw_input or raw_input["gpa"] is None or raw_input["gpa"] == "":
        errors["gpa"] = "This field is required."
    else:
        gpa_value = raw_input["gpa"]
        # Attempt numeric coercion for string inputs
        try:
            gpa_numeric = float(gpa_value)
        except (TypeError, ValueError):
            errors["gpa"] = "GPA must be a numeric value."
            gpa_numeric = None

        if gpa_numeric is not None:
            if gpa_numeric < 0.0 or gpa_numeric > 100.0:
                errors["gpa"] = "GPA must be between 0.0 and 100.0."

    # --- technical_skills ---
    if "technical_skills" not in raw_input or raw_input["technical_skills"] is None:
        errors["technical_skills"] = "This field is required."
    else:
        ts = raw_input["technical_skills"]
        if isinstance(ts, list):
            if len(ts) == 0:
                errors["technical_skills"] = "Technical skills cannot be empty."
        elif isinstance(ts, str):
            if ts.strip() == "":
                errors["technical_skills"] = "Technical skills cannot be empty."
        else:
            # Any other type (int, dict, …) is treated as invalid
            errors["technical_skills"] = "Technical skills cannot be empty."

    # --- interests ---
    if "interests" not in raw_input or raw_input["interests"] is None:
        errors["interests"] = "This field is required."
    else:
        interests = raw_input["interests"]
        if isinstance(interests, list):
            if len(interests) == 0:
                errors["interests"] = "Interests cannot be empty."
        elif isinstance(interests, str):
            if interests.strip() == "":
                errors["interests"] = "Interests cannot be empty."
        else:
            errors["interests"] = "Interests cannot be empty."

    return (len(errors) == 0, errors)
