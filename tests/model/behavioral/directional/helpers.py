"""helpers.py

Module containing tests for verifying the model's directional behavioral
requirements.

These tests ensure the model reacts in a predictable and expected manner when the
local features are perturbed in a specific manner. This confirms the model is learning
the intended directional semantics and is sensitive to event sequence.

Functions:
    model_directional_tests_setup() -> tuple[
        DataLoader,
        Module,
        device,
        PipelineConfig,
        TestsConfig,
    ]
        Pytest fixture initializing the environment for directional tests.
    test_model_directional_local_feature_perturbations(
        model_directional_tests_setup: tuple,
        feature: str
    ) -> None
        Tests the model's directional sensitivity to local features.
"""

import numpy as np
import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    LIST_LAST_IDX,
    TENSOR_CLASS_DIM,
)
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE,
    DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE,
    DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS,
)
from tests.model.helpers import initialize_inference_environment


@pytest.fixture(scope="module")
def model_directional_tests_setup() -> tuple[
    DataLoader,
    Module,
    device,
    PipelineConfig,
    TestsConfig,
]:
    """Pytest fixture that initializes the entire environment
    for directional tests.

    This fixture ensures that the setup (`initialize_inference_environment`)
    runs once per module, providing the necessary components to all directional
    test functions.

    Returns:
        tuple[
            DataLoader,
            Module,
            device,
            PipelineConfig,
            TestsConfig,
        ]:
            - testing_loader: DataLoader for the testing set.
            - model: The loaded, best-performing model instance.
            - device: The device selected for inference.
            - pipeline_config: The pipeline configuration object.
            - tests_config: The tests configuration object.
    """
    return initialize_inference_environment()


@pytest.mark.parametrize(
    "feature",
    DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS,
)
@pytest.mark.model_behavioral_directional_local_feature_perturbations
@pytest.mark.training
def test_model_directional_local_feature_perturbations(
    model_directional_tests_setup: tuple,
    feature: str,
):
    """Tests the model's directional sensitivity to extreme
    local feature perturbations.

    This test verifies that forcing a specific local feature to its maximum value
    at the last timestep leads to a higher prediction probability for the target
    class (the key at the last timestep) compared to forcing the same feature
    to its minimum value.
    The check executes the following steps:
    1.  Identifies the target class for each sequence.
    2.  Creates two perturbed feature sets:
        * High scenario: Forces the feature at the last timestep to its maximum
                         allowed value.
        * Low scenario: Forces the feature at the last timestep to its minimum
                        allowed value.
    3.  Computes the probability of the target class for both the high and low
        scenarios.
    4.  Calculates the success ratio as the fraction of sequences where
        P(high) > P(low).
    5.  Asserts that the success ratio meets or exceeds the minimum required ratio
        defined in the configuration.

    Args:
        model_directional_tests_setup (tuple): Fixture providing the testing loader, model,
                                               device, and tests configuration.
        feature (str): The specific local feature column being tested
                       (injected via parametrization).

    Raises:
        AssertionError: If the success ratio is below the configured minimum required ratio.
    """
    # Setup
    testing_loader, model, _, _, tests_config = model_directional_tests_setup

    # Take the first batch
    batch = next(iter(testing_loader))
    x_features, x_keys, _ = batch

    # Map feature names to their indices
    feature_to_idx = {
        f: i for i, f in enumerate(DATASET_PROCESSED_FEATURE_COLUMNS)
    }
    idx = feature_to_idx[feature]

    # Target classes: the key at the last
    # position of the sequence
    target_classes = x_keys[:, LIST_LAST_IDX]

    # Clone features before perturbation
    x_features_high = x_features.clone()

    # Force the feature to the maximum value
    # only at the last timestep
    x_features_high[:, LIST_LAST_IDX, idx] = (
        DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE
    )

    # Compute probabilities for high scenario
    with torch.no_grad():
        probs_high_full = (
            torch.softmax(
                model(x_features_high, x_keys),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Extract probability of the target
    # class in high scenario
    probs_high = probs_high_full[
        np.arange(len(x_features)),
        target_classes,
    ]

    # Clone features before perturbation
    x_features_low = x_features.clone()

    # Force the feature to the minimum value
    # only at the last timestep
    x_features_low[:, LIST_LAST_IDX, idx] = (
        DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE
    )

    # Compute probabilities for low scenario
    with torch.no_grad():
        probs_low_full = (
            torch.softmax(
                model(x_features_low, x_keys),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Extract probability of the target
    # class in low scenario
    probs_low = probs_low_full[
        np.arange(len(x_features)),
        target_classes,
    ]

    # Check for each sequence if P(high) > P(low) and
    # calculate the fraction of successful cases
    success_mask = probs_high > probs_low
    success_ratio = np.mean(success_mask)

    # Assert that the model respects the feature direction
    # in a sufficient percentage of cases
    assert (
        success_ratio
        >= tests_config.model.behavioral.directional.local_feat_perturbations.min_success_ratio
    )
