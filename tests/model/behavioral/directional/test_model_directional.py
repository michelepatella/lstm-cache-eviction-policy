"""test_model_directional.py

Module containing the main test function for verifying the directional behavioral
requirements of the trained model.

These tests ensure the model's predictions react predictably when specific input
features (local, temporal, or sequence order) are perturbed. This is necessary for
validating that the model has learned the intended directional dependencies and sequence
sensitivity.

Functions:
    test_model_directional() -> None
        Runs the comprehensive test suite to validate the model's directional behavior.
"""

import pytest
from helpers import (  # noqa
    model_directional_tests_setup,
    test_model_directional_local_feature_perturbations,
    test_model_directional_request_swap_perturbations,
    test_model_directional_temporal_feature_perturbations,
)

from tests.const import DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS
from tests.model.helpers import initialize_inference_environment


@pytest.mark.model_behavioral_directional
def test_model_directional() -> None:
    """Runs the directional behavioral tests suite on the trained model.

    This function initializes the environment and executes the following directional
    tests using helper functions:
        1.  Local Feature Perturbations: Tests the model's directional sensitivity
            to extreme feature perturbation by verifying that forcing a local feature
            (frequency or recency) to its maximum value at the last timestep yields
            a higher predicted probability for the target class compared to forcing
            it to its minimum value.
        2.  Temporal Feature Perturbations: Tests the model's reliance on time by
            inverting the temporal encoding features (sin/cos time components, equivalent
            to a 180° time shift) and verifying that this shift causes a significant and
            measurable change in the probabilities of the originally predicted classes.
        3.  Request Swap Perturbations: Tests the model's sequence sensitivity by
            swapping the last and penultimate requests in the input sequence and
            verifying that this structural change causes a significant and measurable
            shift in prediction probabilities.

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
    # Test model directional against
    # local features, including frequency and recency
    for local_feature in DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS:
        test_model_directional_local_feature_perturbations(
            (testing_loader, model, device, pipeline_config, tests_config),
            local_feature,
        )

    # Test model directional against temporal feature
    # perturbations, including sin and cos time
    test_model_directional_temporal_feature_perturbations(
        (testing_loader, model, device, pipeline_config, tests_config),
    )

    # Test model directional against requests swapping
    test_model_directional_request_swap_perturbations(
        (testing_loader, model, device, pipeline_config, tests_config),
    )
