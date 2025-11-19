"""test_model_directional.py

Module containing the main test function for verifying the directional behavioral
requirements of the trained model.

These tests ensure the model's predictions react predictably when specific input
features (like local recency/frequency metrics or temporal encodings) are perturbed
in a controlled direction. This is necessary for validating that the model has learned
the intended directional dependencies.

Functions:
    test_model_directional() -> None
        Runs the tests suite to validate the model's directional behavior.
"""

import pytest
from helpers import (  # noqa
    model_directional_tests_setup,
    test_model_directional_local_feature_perturbations,
    test_model_directional_temporal_feature_perturbations,
)

from tests.const import DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS
from tests.model.helpers import initialize_inference_environment


@pytest.mark.model_behavioral_directional
def test_model_directional() -> None:
    """Runs the directional behavioral tests suite on the trained model.

    This function initializes the environment and executes two main categories of
    directional tests using helper functions:
    -   Local Feature Perturbations: Tests the model's sensitivity by checking
        if perturbing local features (like frequency and recency) in a specific
        direction leads to an expected change in predicted class probabilities.
    -   Temporal Feature Perturbations: Tests the model's reliance on time by
        inverting the temporal encoding features (sin/cos time components, equivalent
        to a 180° time shift) and verifying that this shift causes a significant and
        measurable change in the probabilities of the originally predicted classes.

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
    # Directional Tests
    # ----------------------------
    # Test model directional against local feature perturbations,
    # including local frequency and recency (whether the probabilities
    # of the predicted classes over the original probabilities
    # significatively change when significatively perturbing the
    # local feature)
    for local_feature in DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS:
        test_model_directional_local_feature_perturbations(
            (testing_loader, model, device, pipeline_config, tests_config),
            local_feature,
        )

    # Test model directional against temporal feature
    # perturbations, including sin and cos time (whether
    # the probabilities of the predicted classes over the
    # original probabilities significatively change when
    # shifting the time)
    test_model_directional_temporal_feature_perturbations(
        (testing_loader, model, device, pipeline_config, tests_config),
    )
