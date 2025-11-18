"""test_model_invariance.py

Module containing the test function for verifying the behavioral invariance
of the trained model.

This test ensures that minor, non-critical changes to input data do not lead
to significant or unexpected changes in the model's predictions, thereby
validating the model's robustness and stability.

Functions:
    test_model_invariance() -> None
        Runs checks to validate the model's behavioral invariance.
"""

import pytest

from tests.model.behavioral.invariance.helpers import (
    test_model_invariance_feature_perturbations,
)
from tests.model.helpers import initialize_inference_environment


@pytest.mark.model_behavioral_invariance
def test_model_invariance() -> None:
    """Runs the behavioral invariance test suite on the trained model.

    This function sets up the inference environment and executes specific checks
    to ensure the model's predictions remain stable when faced with minor
    perturbations applied to its input.
    The check performed is:
        - Feature Perturbations: Asserts that prediction outcomes are invariant
          (or change negligibly) after applying small, irrelevant modifications
          to the feature values.

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
    # Invariance Tests
    # ----------------------------
    # Tests model invariance against
    # small, irrelevant feature perturbations
    test_model_invariance_feature_perturbations(
        (testing_loader, model, device, pipeline_config, tests_config),
    )
