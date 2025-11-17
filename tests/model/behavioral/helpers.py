"""helpers.py

Module containing the tests for initializing and verifying the behavioral
invariance of the trained model.

This module focuses on robustness checks by assessing the stability of
model predictions when small, irrelevant perturbations are intentionally
introduced into the input features. It uses a Pytest fixture for setup and a
test function to validate that prediction probabilities remain within defined
absolute and relative tolerance thresholds, thereby confirming that the model's
behavior is stable and invariant to minor feature changes.

Functions:
    model_invariance_tests_setup() -> tuple[
        DataLoader,
        Module,
        device,
        PipelineConfig,
        TestsConfig,
    ]
        Pytest fixture initializing the inference environment for invariance
        tests.
    test_model_invariance_feature_perturbations(
        model_invariance_tests_setup: tuple
    ) -> None
        Tests the model's invariance against small perturbations applied to
        input features.
"""

import numpy as np
import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import TENSOR_CLASS_DIM
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from tests.config.pydantic.tests_config import TestsConfig
from tests.model.helpers import initialize_inference_environment


@pytest.fixture(scope="module")
def model_invariance_tests_setup() -> tuple[
    DataLoader,
    Module,
    device,
    PipelineConfig,
    TestsConfig,
]:
    """Pytest fixture that initializes the entire environment
    for invariance tests.

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


def test_model_invariance_feature_perturbations(
    model_invariance_tests_setup: tuple,
):
    """Tests model invariance against small, irrelevant feature perturbations.

    This test verifies that the model's prediction probabilities do not change
    significantly when a small, non-critical perturbation is applied to each
    feature and timestep of the input sequence.

    The test compares the original probabilities against the perturbed probabilities,
    asserting that the relative or absolute difference between them is below
    the predefined tolerances for all predictions.

    Args:
        model_invariance_tests_setup (tuple): Fixture containing the initialized
                                              inference environment and configurations.

    Raises:
        AssertionError: If the prediction probabilities after perturbation differ
                        significantly from the original probabilities, exceeding
                        the configured tolerance thresholds.
    """
    # Setup
    (testing_loader, model, _, pipeline_config, tests_config) = (
        model_invariance_tests_setup
    )

    # Take the first batch from testing loader
    batch = next(iter(testing_loader))
    x_features, x_keys, _ = batch

    # Calculate the original model probabilities
    # (on the original features) over a single batch
    with torch.no_grad():
        original_probs = (
            torch.softmax(model(x_features, x_keys), dim=TENSOR_CLASS_DIM)
            .detach()
            .cpu()
            .numpy()
        )

    # Applying a small perturbation to all the
    # features of the sequence requests, the resulting model
    # probabilities should be similar to the original ones
    _, _, num_features = x_features.shape
    for timestep in range(pipeline_config.model.sequence.lenght):
        for features_idx in range(num_features):
            # Apply a small perturbation to the
            # current feature for all the sequence requests
            x_features_perturbed = x_features.clone()
            x_features_perturbed[:, timestep, features_idx] += (
                tests_config.model.behavioral.invariance.feat_perturbation
            )

            # Calculate model probabilities over
            # perturbed features
            perturbed_probs = (
                torch.softmax(
                    model(x_features_perturbed, x_keys),
                    dim=TENSOR_CLASS_DIM,
                )
                .detach()
                .cpu()
                .numpy()
            )

            # Assert that the probabilities obtained over
            # the perturbed features are similar enough to
            # the original ones, considering both the relative
            # and absolute difference thereof
            abs_diff = np.abs(original_probs - perturbed_probs)
            rel_diff = abs_diff / original_probs
            assert np.all(
                (abs_diff < tests_config.model.behavioral.invariance.abs_tol)
                | (
                    rel_diff < tests_config.model.behavioral.invariance.rel_tol
                ),
            )
