"""
Pytest configuration and shared fixtures for the Career Path Prediction test suite.

Registers the 'ci' Hypothesis profile with max_examples=100 and loads it
so that all property-based tests in this suite run with consistent settings.
"""

from hypothesis import settings

# Register and activate the CI profile so every property test runs
# at least 100 examples.
settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")
