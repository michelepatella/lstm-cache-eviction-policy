"""test_model_minimum_functionality.py

Module containing the test function for verifying the minimum functional
requirements of the trained model.

These tests ensure the model is operating as a predictive system by checking
against two common pitfalls:
    1.  Feature influence: Ensuring the model actually uses the input features
                           (not just the keys).
    2.  Min-Max Probs Gap: Ensuring the model produces sharp, distinguishable
                           probability outputs (not a flat/uniform distribution,
                           which indicates random guessing).

Functions:
    test_model_minimum_functionality() -> None
        Runs the comprehensive test suite to validate the model's minimum
        functionality requirements.
"""

import pytest

from tests.model.behavioral.minimum_functionality.helpers import (
    test_model_minimum_functionality_features_influence,
    test_model_minimum_functionality_min_max_probs_gap,
)
from tests.model.helpers import initialize_inference_environment


@pytest.mark.model_behavioral_minimum_functionality
def test_model_minimum_functionality() -> None:
    """Runs the minimum functionality behavioral test suite on
    the trained model.

    This function initializes the environment and executes helper tests to ensure
    the model meets predictive standards:
        -   Features influence: Ensures that zeroing out the input features
            (local recency, frequency, time) causes a measurable shift in the
            model's output probabilities, confirming that the model uses the features
            and hasn't learned a 'key-only' shortcut.
        -   Min-max probabilities gap: Verifies that the difference between the
            maximum and minimum output probabilities is sufficiently large, confirming
            the model exhibits decision clarity and does not produce a uniform
            (random guessing) distribution.

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    # Initialize the inference environment
    (testing_loader, model, device, pipeline_config, tests_config) = (
        initialize_inference_environment()
    )

    # ----------------------------
    # Minimum Functionality Tests
    # ----------------------------
    # Test model minimum functionality by checking
    # features have influence on the model's predictions
    test_model_minimum_functionality_features_influence(
        (testing_loader, model, device, pipeline_config, tests_config),
    )

    # Test model minimum functionality by checking
    # there's enough gap among min-max probabilities
    test_model_minimum_functionality_min_max_probs_gap(
        (testing_loader, model, device, pipeline_config, tests_config),
    )
