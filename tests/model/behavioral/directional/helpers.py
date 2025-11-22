"""helpers.py

Module containing tests for verifying the model's directional behavioral
requirements.

These tests ensure the model reacts in a predictable and expected manner when the
local features, temporal features, or sequence order are perturbed in a specific manner.
This confirms the model is learning the intended directional semantics and is sensitive
to event sequence.

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
    test_model_directional_temporal_feature_perturbations(
        model_directional_tests_setup: tuple
    ) -> None
        Tests the model's directional sensitivity to temporal feature inversion
        (180° shift).
    test_model_directional_request_swap_perturbations(
        model_directional_tests_setup: tuple
    ) -> None
        Tests the model's sensitivity to sequence order by swapping the last two
        elements (requests and features).
"""

import numpy as np
import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import (
    DATASET_COLUMN_COS_TIME_NAME,
    DATASET_COLUMN_SIN_TIME_NAME,
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
    L1_MAX_DISTANCE,
    MODEL_BEHAVIORAL_DIRECTIONAL_TESTS_LOCAL_FEATURE_PERTURBATIONS_PARAM_FEATURE_NAME,
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
    MODEL_BEHAVIORAL_DIRECTIONAL_TESTS_LOCAL_FEATURE_PERTURBATIONS_PARAM_FEATURE_NAME,
    DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS,
)
@pytest.mark.model_behavioral_directional_local_feature_perturbations
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


@pytest.mark.model_behavioral_directional_temporal_feature_perturbations
def test_model_directional_temporal_feature_perturbations(
    model_directional_tests_setup: tuple,
):
    """Tests the model's directional sensitivity to temporal feature
    perturbations.

    This test verifies that inverting the temporal components of the input
    features, which corresponds to a 180-degree shift in time, causes a
    significant and measurable change in the entire model's prediction
    probability distribution, thereby confirming the model's reliance on
    temporal directionality.
    The test executes the following steps:
        1.  Computes the original prediction probabilities over a batch.
        2.  Applies a perturbation by inverting the sin and cos time feature values
            across all timesteps and sequences (achieving a 180° phase shift).
        3.  Computes the new probabilities on the altered features.
        4.  Calculates the average L1 distance (sum of absolute differences) between
            the original probability distribution and the inverted probability
            distribution across the batch.
        5.  Assert that this average global shift meets or exceeds the minimum
            required shift defined in the configuration.

    Args:
        model_directional_tests_setup (tuple): Fixture providing the testing loader,
                                               model, device, and tests configuration.

    Raises:
        AssertionError: If the average L1 distance between the probability
                        distributions is below the configured minimum required shift.
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

    # Get sin and cos time indexes
    sin_time_idx = feature_to_idx[DATASET_COLUMN_SIN_TIME_NAME]
    cos_time_idx = feature_to_idx[DATASET_COLUMN_COS_TIME_NAME]

    # Compute original probabilities
    with torch.no_grad():
        original_probs = (
            torch.softmax(
                model(x_features, x_keys),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Clone features for perturbation
    x_features_inverted = x_features.clone()

    # Apply perturbation to all the element in the
    # batch sequences, inverting both sin(t) and cos(t)
    # components (180 degrees shift)
    x_features_inverted[:, :, cos_time_idx] = -x_features_inverted[
        :,
        :,
        cos_time_idx,
    ]
    x_features_inverted[:, :, sin_time_idx] = -x_features_inverted[
        :,
        :,
        sin_time_idx,
    ]

    # Compute probabilities on altered features
    with torch.no_grad():
        inverted_probs = (
            torch.softmax(
                model(x_features_inverted, x_keys),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Calculate the L1 distance between the original
    # and altered probability distributions for
    # each sequence in the batch
    l1_distances = (
        np.sum(
            np.abs(original_probs - inverted_probs),
            axis=TENSOR_CLASS_DIM,
        )
        / L1_MAX_DISTANCE
    )

    # Assert that the entire probability distribution has
    # been consistently altered after having shifted the time,
    # by checking the average L1 distance across the batch
    avg_shift = np.mean(l1_distances)
    assert (
        avg_shift
        >= tests_config.model.behavioral.directional.temporal_feat_perturbations.min_shift
    )


@pytest.mark.model_behavioral_directional_request_swap_perturbations
def test_model_directional_request_swap_perturbations(
    model_directional_tests_setup: tuple,
):
    """Tests the model's sensitivity to sequence order by swapping the last two
    elements.

    This test verifies that swapping the position of the last and penultimate
    keys (and their corresponding features) in the input sequence causes a
    significant and measurable change in the entire model's prediction
    probability distribution. This confirms the model relies on the specific
    sequential order of recent events.
    The test executes the following steps:
        1.  Computes the original prediction probabilities over a batch.
        2.  Applies a perturbation by swapping the features and keys of the last
            time step with those of the penultimate time step for all sequences.
        3.  Computes the new probabilities on the altered sequences.
        4.  Calculates the average L1 distance (sum of absolute differences) between
            the original probability distribution and the altered probability
            distribution across the batch.
        5.  Assert that this average global shift meets or exceeds the minimum
            required shift defined in the configuration.

    Args:
        model_directional_tests_setup (tuple): Fixture providing the testing loader,
                                               model, device, and tests configuration.

    Raises:
        AssertionError: If the average L1 distance between the probability
                        distributions is below the configured minimum required shift.
    """
    # Setup
    testing_loader, model, _, _, tests_config = model_directional_tests_setup

    # Take the first batch
    batch = next(iter(testing_loader))
    x_features, x_keys, _ = batch

    # Compute original probabilities
    with torch.no_grad():
        original_probs = (
            torch.softmax(
                model(x_features, x_keys),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Clone features and keys for perturbation
    x_features_swapped = x_features.clone()
    x_keys_swapped = x_keys.clone()

    # Indices for the last and penultimate elements
    last_idx = LIST_LAST_IDX
    penultimate_idx = LIST_LAST_IDX - 1

    # Apply perturbation: Swap the last and penultimate elements
    # (features and keys) for all sequences in the batch
    temp_features = x_features_swapped[:, penultimate_idx, :].clone()
    temp_keys = x_keys_swapped[:, penultimate_idx].clone()
    # Overwrite penultimate with last
    x_features_swapped[:, penultimate_idx, :] = x_features_swapped[
        :,
        last_idx,
        :,
    ]
    x_keys_swapped[:, penultimate_idx] = x_keys_swapped[:, last_idx]
    # Overwrite last with stored penultimate
    x_features_swapped[:, last_idx, :] = temp_features
    x_keys_swapped[:, last_idx] = temp_keys

    # Compute probabilities on altered sequences
    with torch.no_grad():
        swapped_probs = (
            torch.softmax(
                model(x_features_swapped, x_keys_swapped),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Calculate the L1 distance between the original
    # and altered probability distributions for
    # each sequence in the batch
    l1_distances = (
        np.sum(
            np.abs(original_probs - swapped_probs),
            axis=TENSOR_CLASS_DIM,
        )
        / L1_MAX_DISTANCE
    )

    # Assert that the entire probability distribution has
    # been consistently altered after having swapped the
    # order of the last two elements, by checking the
    # average L1 distance across the batch
    avg_shift = np.mean(l1_distances)
    assert (
        avg_shift
        >= tests_config.model.behavioral.directional.request_swap_perturbations.min_shift
    )
