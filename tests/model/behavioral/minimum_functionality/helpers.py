"""helpers.py

Module containing tests to verify the minimum functional requirements
of the trained model.

These tests ensure the model is operating as a predictive, feature-aware system
and not relying on trivial shortcuts or generating degenerate output distributions.

Functions:
    model_minimum_functionality_tests_setup() -> tuple[
        DataLoader,
        Module,
        device,
        PipelineConfig,
        TestsConfig,
    ]
        Pytest fixture initializing the environment for minimum
        functionality tests.
    test_model_minimum_functionality_features_influence(
        model_minimum_functionality_tests_setup
    ) -> None
        Tests the measurable influence of input features on the model's
        output probabilities.
    test_model_minimum_functionality_min_max_probs_gap(
        model_minimum_functionality_tests_setup
    ) -> None
        Tests the decision clarity of the model by checking the gap
        between minimum and maximum output probabilities.
"""

import numpy as np
import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import (
    TENSOR_CLASS_DIM,
)
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    L1_MAX_DISTANCE,
)
from tests.model.helpers import initialize_inference_environment


@pytest.fixture(scope="module")
def model_minimum_functionality_tests_setup() -> tuple[
    DataLoader,
    Module,
    device,
    PipelineConfig,
    TestsConfig,
]:
    """Pytest fixture that initializes the entire environment
    for minimum functionality tests.

    This fixture ensures that the setup (`initialize_inference_environment`)
    runs once per module, providing the necessary components to all invariance
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


@pytest.mark.model_behavioral_minimum_functionality_features_influence
def test_model_minimum_functionality_features_influence(
    model_minimum_functionality_tests_setup: tuple,
):
    """Tests the influence of features on the model's output.

    This test detects if the model is taking a 'shortcut' by ignoring the
    input features (local frequency, recency, time) and relying solely on
    the key sequence embeddings.
    The test executes the following steps:
        1.  Computes probabilities for a standard batch.
        2.  Creates a synthetic batch where the features are zeroed out but
            the keys remain identical.
        3.  Computes probabilities for this featureless batch.
        4.  Calculates the average L1 distance between original and featureless
            outputs.
        5.  Asserts that the distance is greater than a configured value.

    Args:
        model_minimum_functionality_tests_setup (tuple): Fixture providing the
                                                         testing loader, model,
                                                         device, and tests
                                                         configuration.

    Raises:
        AssertionError: If the L1 distance is effectively zero, indicating the
                        model ignores the features.
    """
    # Setup
    testing_loader, model, _, _, tests_config = (
        model_minimum_functionality_tests_setup
    )

    # Take the first batch
    batch = next(iter(testing_loader))
    x_features, x_keys, _ = batch

    # Compute probabilities for the
    # original batch
    with torch.no_grad():
        original_probs = torch.softmax(
            model(x_features, x_keys),
            dim=TENSOR_CLASS_DIM,
        )

    # Create a featureless input set keeping
    # the keys identical but zero out all features
    x_features_zeroed = torch.zeros_like(x_features)

    # Compute probabilities for the
    # featureless batch
    with torch.no_grad():
        featureless_probs = torch.softmax(
            model(x_features_zeroed, x_keys),
            dim=TENSOR_CLASS_DIM,
        )

    # Calculate the L1 distance between the
    # two probability distributions
    l1_distances = (
        np.sum(
            np.abs(
                original_probs.cpu().numpy() - featureless_probs.cpu().numpy(),
            ),
            axis=TENSOR_CLASS_DIM,
        )
        / L1_MAX_DISTANCE
    )

    # Calculate the average shift across the batch
    avg_influence_shift = np.mean(l1_distances)

    # Assert that the features have
    # a measurable impact on the prediction
    assert (
        avg_influence_shift
        > tests_config.model.behavioral.minimum_functionality.feat_influence.min_probs_shift
    )


@pytest.mark.model_behavioral_minimum_functionality_min_max_probs_gap
def test_model_minimum_functionality_min_max_probs_gap(
    model_minimum_functionality_tests_setup: tuple,
):
    """Tests the model's decision-making capability.

    This test checks the decision-making capability of the model. A functional
    predictive model must distinguish between likely and unlikely events. If
    the model outputs a flat distribution (where the most likely class is barely
    distinguishable from the least likely), it has failed to learn a decision
    boundary (random guessing shortcut).
    The test executes the following steps:
        1.  Computes softmax probabilities for a batch.
        2.  Calculates the global maximum and minimum probabilities.
        3.  Asserts that the difference among these probabilities is greater
            than a configured value.

    Args:
        model_minimum_functionality_tests_setup (tuple): Fixture providing the
                                                         testing loader, model,
                                                         device, and tests
                                                         configuration.

    Raises:
        AssertionError: If the probabilities gap is too small, indicating a
                        degenerate or uniform output distribution.
    """
    # Setup
    testing_loader, model, _, _, tests_config = (
        model_minimum_functionality_tests_setup
    )

    # Take the first batch
    batch = next(iter(testing_loader))
    x_features, x_keys, _ = batch

    # Compute probabilities
    with torch.no_grad():
        probs = torch.softmax(
            model(x_features, x_keys),
            dim=TENSOR_CLASS_DIM,
        )

    # Calculate the global maximum and minimum
    # probability values across the batch
    max_prob = torch.max(probs).item()
    min_prob = torch.min(probs).item()

    # Calculate the probabilities gap
    probs_gap = max_prob - min_prob

    # Assert that the model shows decision clarity
    assert (
        probs_gap
        > tests_config.model.behavioral.minimum_functionality.min_max_probs_gap.min
    )
