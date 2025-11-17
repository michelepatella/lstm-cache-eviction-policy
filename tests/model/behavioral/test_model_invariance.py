"""test_model_invariance.py

Module containing the test function for verifying the behavioral invariance
of the trained model against small, irrelevant perturbations in its input features.

This test ensures that minor, non-critical changes to input data (feature
perturbations) do not lead to significant or unexpected changes in the model's
predictions, thereby validating the model's robustness and stability.

Functions:
    test_model_invariance() -> None
        Runs checks to validate the model's behavioral invariance.
"""

from tests.model.behavioral.helpers import (
    model_invariance_tests_setup,  # noqa
    test_model_invariance_feature_perturbations,
)
from tests.model.helpers import initialize_inference_environment


def test_model_invariance() -> None:
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
