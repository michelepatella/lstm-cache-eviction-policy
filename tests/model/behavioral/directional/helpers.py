"""helpers.py

Module containing tests for verifying the model's directional behavioral
requirements.

These tests ensure the model reacts in a predictable and expected manner when the
temporal encoding, or sequence order are perturbed in a specific manner, as well as
the tests verify the consistency of the model with respect to the local features (recency
and frequency). This confirms the model is learning the intended directional semantics
and is sensitive to event sequence.

Functions:
    model_directional_tests_setup() -> tuple[
        DataLoader,
        Module,
        device,
        PipelineConfig,
        TestsConfig,
    ]
        Pytest fixture initializing the environment for directional tests.
    test_model_directional_local_feature_consistency(
        model_directional_tests_setup: tuple,
        feature: str
    ) -> None
        Tests the model's consistency with respect to local features.
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
    TENSOR_FEATURES_DIM,
)
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS,
    L1_MAX_DISTANCE,
    MODEL_BEHAVIORAL_DIRECTIONAL_TESTS_LOCAL_FEATURE_CONSISTENCY_PARAM_FEATURE_NAME,
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
    MODEL_BEHAVIORAL_DIRECTIONAL_TESTS_LOCAL_FEATURE_CONSISTENCY_PARAM_FEATURE_NAME,
    DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS,
)
@pytest.mark.model_behavioral_directional_local_feature_consistency
def test_model_directional_local_feature_consistency(
    model_directional_tests_setup: tuple,
    feature: str,
):
    """Tests the model's directional consistency of local features.

    This test verifies that the model naturally assigns higher (or equal)
    prediction probability to keys associated with higher values of local
    features (frequency/recency) compared to keys with lower values within
    the same sequence. The check executes the following steps:
        1.  Computes the prediction probabilities for the batch using
            original inputs.
        2.  Identifies the timestep where the feature value is maximized.
        3.  Identifies the timestep where the feature value is minimized.
        4.  Extracts the keys associated with these timestamps.
        5.  Compares the predicted probabilities for these specific keys.
        6.  Asserts that the average probability of the 'high-feature key' is
            greater than or equal to the 'low-feature key' by a configured margin.

    Args:
        model_directional_tests_setup (tuple): Fixture providing the testing loader,
                                               model, device, and tests configuration.
        feature (str): The specific local feature column being tested
                       (injected via parametrization).

    Raises:
        AssertionError: If the model assigns higher probability to low-feature
                        keys on average, contradicting the expected directional
                        behavior.
    """
    # Setup
    testing_loader, model, device, _, tests_config = (
        model_directional_tests_setup
    )

    # Take the first batch
    batch = next(iter(testing_loader))
    x_features, x_keys, _ = batch

    # Map feature names to their indices
    feature_to_idx = {
        f: i for i, f in enumerate(DATASET_PROCESSED_FEATURE_COLUMNS)
    }
    idx = feature_to_idx[feature]

    # Compute prediction probabilities on the original
    # unaltered input sequence
    with torch.no_grad():
        prediction_probs = (
            torch.softmax(
                model(x_features, x_keys),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Extract the specific feature values
    # for all timesteps
    sequence_feature_values = x_features[:, :, idx]

    # Identify the timestep indices with the maximum
    # and minimum feature values for each sequence
    t_high_indices = torch.argmax(
        sequence_feature_values,
        dim=TENSOR_FEATURES_DIM,
    )
    t_low_indices = torch.argmin(
        sequence_feature_values,
        dim=TENSOR_FEATURES_DIM,
    )

    # Create a range for batch indexing
    batch_indices = torch.arange(len(x_features), device=device)

    # Extract the keys associated with the high
    # and low timesteps
    high_feature_keys = x_keys[batch_indices, t_high_indices]
    low_feature_keys = x_keys[batch_indices, t_low_indices]

    # Extract the predicted probabilities for
    # the identified keys
    high_feature_keys_probs = prediction_probs[
        np.arange(len(x_features)),
        high_feature_keys.cpu().numpy(),
    ]
    low_feature_keys_probs = prediction_probs[
        np.arange(len(x_features)),
        low_feature_keys.cpu().numpy(),
    ]

    # Calculate the average difference
    avg_prob_difference = np.mean(
        high_feature_keys_probs - low_feature_keys_probs,
    )

    # Assert that the key with the higher feature value
    # has, on average, a higher predicted probability than
    # the key with the lower feature value
    assert (
        avg_prob_difference
        >= tests_config.model.behavioral.directional.local_feat_perturbations.min_shift
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
