"""helpers.py

Module containing tests for verifying the model's directional behavioral
requirements.

These tests ensure the model reacts in a predictable and expected manner when the
features (those related to local recency/frequency or temporal encoding) are
perturbed in a specific direction. This confirms the model is learning the intended
directional semantics from the data.

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
        Tests the model's directional sensitivity to local feature scaling.
    test_model_directional_temporal_feature_perturbations(
        model_directional_tests_setup: tuple
    ) -> None
        Tests the model's directional sensitivity to temporal feature inversion
        (180° shift).
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
    """Tests the model's sensitivity to local feature scaling by measuring
    probability shift.

    This test verifies that altering a local feature (recency or frequency)
    in the last element of a sequence causes a significant shift in the
    prediction probability for the originally predicted class. This ensures the
    model is sensitive to recent, local context changes.
    The check is parameterized over local feature columns and executes the
    following steps:
    1.  Computes the original probabilities and identifies the predicted class
        for each sequence.
    2.  Applies a proportional perturbation to the selected local feature in the
        last element of all sequences, clamping the result within bounds.
    3.  Computes the new probabilities on the perturbed features.
    4.  Calculates the average absolute shift between the original probability and
        the perturbed probability for the originally predicted class.
    5.  Asserts that this average shift meets or exceeds the minimum required shift
        defined in the configuration.

    Args:
        model_directional_tests_setup (tuple): Fixture providing the testing loader,
                                               model, device, and tests configuration.
        feature (str): The specific local feature column being tested
                       (injected via parametrization).

    Raises:
        AssertionError: If the average absolute probability shift caused by the local
                        feature perturbation is below the configured minimum required
                        shift.
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

    # Take the predicted classes over the original features
    # and extract their probabilities
    original_pred_classes = np.argmax(original_probs, axis=TENSOR_CLASS_DIM)
    original_pred_class_probs = original_probs[
        np.arange(len(x_features)),
        original_pred_classes,
    ]

    # Clone features for perturbation
    x_features_perturbed = x_features.clone()

    # Alter the feature of the last element
    # in all the batch sequences, ensuring
    # it still lies within the predefined range
    perturbed_values = (
        x_features_perturbed[:, LIST_LAST_IDX, idx]
        * tests_config.model.behavioral.directional.local_feat_perturbations.level
    )
    x_features_perturbed[:, LIST_LAST_IDX, idx] = torch.clamp(
        perturbed_values,
        min=DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE,
        max=DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE,
    )

    # Compute probabilities on perturbed features
    with torch.no_grad():
        perturbed_probs = (
            torch.softmax(
                model(x_features_perturbed, x_keys),
                dim=TENSOR_CLASS_DIM,
            )
            .cpu()
            .numpy()
        )

    # Extract the perturbed probabilities of the
    # predicted classes over the original features
    perturbed_pred_class_probs = perturbed_probs[
        np.arange(len(x_features)),
        original_pred_classes,
    ]

    # Assert that the probabilities of the predicted
    # classes over the original features have been
    # significatively changed after altering the local
    # feature of the last element in the sequence
    avg_shift = np.mean(
        abs(original_pred_class_probs - perturbed_pred_class_probs),
    )
    assert (
        avg_shift
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
    significant and measurable change in the model's prediction probabilities,
    thereby confirming the model's reliance on temporal directionality.

    The test executes the following steps:
    1.  Computes the original prediction probabilities over a batch.
    2.  Identifies the class predicted with the highest probability for each
        sequence.
    3.  Applies a perturbation by inverting the sin and cos time feature values
        across all timesteps and sequences (achieving a 180° phase shift).
    4.  Computes the new probabilities on the altered features.
    5.  Calculates the average absolute difference between the original probability
        and the inverted probability for the originally predicted class.
    6.  Assert that this average shift meets or exceeds the minimum required shift
        defined in the configuration.

    Args:
        model_directional_tests_setup (tuple): Fixture providing the testing loader,
                                               model, device, and tests configuration.

    Raises:
        AssertionError: If the average absolute probability shift caused by the temporal
                        inversion is below the configured minimum required shift.
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

    # Take the predicted classes (with the highest probabilities)
    # over the original features and extract their probabilities
    # from all the batch sequences
    original_pred_classes = np.argmax(original_probs, axis=TENSOR_CLASS_DIM)
    original_pred_class_probs = original_probs[
        np.arange(len(x_features)),
        original_pred_classes,
    ]

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

    # Extract the altered probabilities of the
    # classes predicted over the original features
    inverted_pred_class_probs = inverted_probs[
        np.arange(len(x_features)),
        original_pred_classes,
    ]

    # Assert that the altered probabilities of
    # the predicted classes over the original
    # features have been consistently altered
    # after having shifted the time
    avg_shift = np.mean(
        abs(original_pred_class_probs - inverted_pred_class_probs),
    )
    assert (
        avg_shift
        >= tests_config.model.behavioral.directional.temporal_feat_perturbations.min_shift
    )
