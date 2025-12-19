"""test_model_invariance.py

Module containing the test function for verifying the behavioral invariance
of the trained model.

This test ensures that minor, non-critical changes to input data do not lead
to significant or unexpected changes in the model's predictions, thereby
validating the model's robustness, stability, and deterministic behavior.

Functions:
    test_model_invariance() -> None
        Runs checks to validate the model's behavioral invariance.
"""

import pytest

from components.const import DATASET_PROCESSED_FEATURE_COLUMNS
from tests.model.behavioral.invariance.helpers import (
    test_model_invariance_feature_perturbation,
    test_model_invariance_identity_preservation,
)
from tests.model.helpers import initialize_inference_environment


@pytest.mark.model_behavioral_invariance
@pytest.mark.training
def test_model_invariance() -> None:
    """Runs the behavioral invariance test suite on the trained model.

    This function sets up the inference environment and executes specific checks
    to ensure the model's predictions remain stable when faced with minor
    perturbations or redundant input.
    The checks performed are:
        - Feature perturbations: Asserts that prediction outcomes are invariant
          (or change negligibly) after applying small, irrelevant modifications
          to the feature values.
        - Identity preservation: Asserts that providing duplicated input data
          yields identical predictions, validating the model's deterministic
          behavior.

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
    # Test model invariance against
    # small, irrelevant feature perturbations
    for feature in DATASET_PROCESSED_FEATURE_COLUMNS:
        feature_idx = DATASET_PROCESSED_FEATURE_COLUMNS.index(feature)
        test_model_invariance_feature_perturbation(
            feature_idx,
            (testing_loader, model, device, pipeline_config, tests_config),
        )

    # Test model invariance against duplicated inputs
    test_model_invariance_identity_preservation(
        (testing_loader, model, device, pipeline_config, tests_config),
    )
