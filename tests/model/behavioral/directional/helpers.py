import numpy as np
import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    LIST_FIRST_IDX,
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
def test_model_directional_local_feature_perturbations(
    model_directional_tests_setup: tuple,
    feature: str,
):
    """Tests the model's directional expectation upon local
    feature perturbations.

    This test verifies that the model's prediction probability for the
    target class changes directionally when a local, relevant feature
    is slightly increased compared to when it is decreased.
    The check is parameterized over local feature columns and executes
    the following steps:
    1.  Applies both a decreased and an increased perturbation level to the
        first element of the current local feature column for all sequences
        in the batch, clamping the values within predefined min/max bounds.
    2.  Computes the probability assigned to the true target class for both
        the decreased and increased feature sets.
    3.  Calculates the success fraction.
    5.  Asserts that the success fraction meets or exceeds the minimum required
        ratio defined in the configuration.

    Args:
        model_directional_tests_setup (tuple): Fixture providing the testing loader,
                                               model, device, and tests configuration.
        feature (str): The specific local feature column being tested (injected via
                       parametrization).

    Raises:
        AssertionError: If the success fraction (proportion of sequences where the
                        increased feature led to a higher probability for the target
                        class than the decreased feature) is below the configured
                        minimum success ratio.
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

    # Target classes: first element of each
    # batch sequences
    target_classes = x_keys[:, LIST_FIRST_IDX]

    # Clone features for decreased/increased
    # perturbations
    x_features_decreased = x_features.clone()
    x_features_increased = x_features.clone()

    # Apply perturbations
    for i in range(len(x_features)):
        # Decrease first element of each
        # batch sequences
        x_features_decreased[i, LIST_FIRST_IDX, idx] = torch.clamp(
            x_features_decreased[i, LIST_FIRST_IDX, idx]
            * (
                DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE
                - tests_config.model.behavioral.directional.local_feat_perturbations.level
            ),
            min=DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE,
            max=DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE,
        )

        # Increase first element of each
        # batch sequences
        x_features_increased[i, LIST_FIRST_IDX, idx] = torch.clamp(
            x_features_increased[i, LIST_FIRST_IDX, idx]
            * (
                DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE
                + tests_config.model.behavioral.directional.local_feat_perturbations.level
            ),
            min=DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MIN_VALUE,
            max=DATASET_COLUMN_LOCAL_FREQUENCY_RECENCY_MAX_VALUE,
        )

    # Compute probabilities on
    # altered features
    with torch.no_grad():
        probs_decreased = (
            torch.softmax(
                model(x_features_decreased, x_keys),
                dim=TENSOR_CLASS_DIM,
            )[torch.arange(len(x_features)), target_classes]
            .cpu()
            .numpy()
        )
        probs_increased = (
            torch.softmax(
                model(x_features_increased, x_keys),
                dim=TENSOR_CLASS_DIM,
            )[torch.arange(len(x_features)), target_classes]
            .cpu()
            .numpy()
        )

    # Assert directional effect: increased > decreased,
    # calculating a success fraction over all the batch
    # sequences
    success_mask = probs_increased >= probs_decreased
    success_fraction = np.mean(success_mask)
    assert (
        success_fraction
        >= tests_config.model.behavioral.directional.local_feat_perturbations.min_success_ratio
    )
