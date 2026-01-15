"""helpers.py

Module containing the tests for initializing and verifying the behavioral
invariance of the trained model.

This module focuses on robustness checks by assessing the stability of
model predictions when small, irrelevant perturbations are intentionally
introduced into the input features, and by verifying its deterministic behavior.

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
    test_model_invariance_feature_perturbation(
        model_invariance_tests_setup: tuple
    ) -> None
        Tests the model's invariance against small perturbations applied to
        input features.
    test_model_invariance_identity_preservation(
        model_invariance_tests_setup: tuple
    ) -> None
        Tests the model's deterministic behavior by verifying identity preservation
        when presented with duplicate input data.
"""

import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    TENSOR_CLASS_DIM,
)
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    TORCH_TOP_K_LARGEST,
    TORCH_TOP_K_SMALLEST,
)
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


@pytest.mark.model_behavioral_invariance_feature_perturbation
@pytest.mark.parametrize(
    "feature_idx",
    range(len(DATASET_PROCESSED_FEATURE_COLUMNS)),
)
@pytest.mark.after_model_training
def test_model_invariance_feature_perturbation(
    feature_idx: int,
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
        feature_idx (int): The index of the feature being tested.
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
        )

    # Applying a small perturbation to all the
    # features of the sequence requests, the resulting model
    # probabilities should be similar to the original ones
    for timestep in range(pipeline_config.model.sequence.length):
        # Apply a small perturbation to the
        # current feature for all the sequence requests
        x_features_perturbed = x_features.clone()
        x_features_perturbed[:, timestep, feature_idx] += (
            tests_config.model.behavioral.invariance.feat_perturbation.level
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
        )

        # Assert that the probabilities obtained over
        # the perturbed features are similar enough to
        # the original ones, considering both the relative
        # and absolute difference thereof
        abs_diff = torch.abs(original_probs - perturbed_probs)
        rel_diff = abs_diff / original_probs
        assert torch.all(
            (
                abs_diff
                < tests_config.model.behavioral.invariance.feat_perturbation.max_probs_shift.abs
            )
            | (
                rel_diff
                < tests_config.model.behavioral.invariance.feat_perturbation.max_probs_shift.rel
            ),
        )

        extreme_k = tests_config.model.behavioral.invariance.feat_perturbation.extreme_k
        # Head probabilities
        _, original_head = torch.topk(
            original_probs,
            k=extreme_k,
            dim=TENSOR_CLASS_DIM,
            largest=TORCH_TOP_K_LARGEST,
        )
        _, perturbed_head = torch.topk(
            perturbed_probs,
            k=extreme_k,
            dim=TENSOR_CLASS_DIM,
            largest=TORCH_TOP_K_LARGEST,
        )

        # Tail probabilities
        _, original_tail = torch.topk(
            original_probs,
            k=extreme_k,
            dim=TENSOR_CLASS_DIM,
            largest=TORCH_TOP_K_SMALLEST,
        )
        _, perturbed_tail = torch.topk(
            perturbed_probs,
            k=extreme_k,
            dim=TENSOR_CLASS_DIM,
            largest=TORCH_TOP_K_SMALLEST,
        )

        # Assert that extreme-k probabilities obtained
        # over perturbed features are the same as
        # those obtained over original features
        assert torch.equal(original_head, perturbed_head)
        assert torch.equal(original_tail, perturbed_tail)


@pytest.mark.model_behavioral_invariance_identity
@pytest.mark.after_model_training
def test_model_invariance_identity_preservation(
    model_invariance_tests_setup: tuple,
):
    """Tests the model's deterministic behavior (identity preservation).

    This test verifies that the model is not acting randomly and
    preserves consistency when presented with the exact same input data.
    Inputting a duplicate of the original sequence must yield the same
    predictions.
    The test executes the following steps:
        1.  Computes probabilities for a standard batch.
        2.  Creates a clone (duplicate) of the input batch features and keys.
        3.  Computes probabilities for this cloned batch.
        4.  Calculates the maximum absolute difference between the two
            output distributions.
        5.  Asserts that the difference is effectively near to zero.

    Args:
        model_invariance_tests_setup (tuple): Fixture providing the
                                              testing loader, model,
                                              device, and tests
                                              configuration.

    Raises:
        AssertionError: If the outputs differ, indicating non-deterministic
                        behavior or instability.
    """
    # Setup
    testing_loader, model, _, _, tests_config = model_invariance_tests_setup

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

    # Create a duplicate input set
    # by cloning the tensors to ensure
    # memory independence
    x_features_clone = x_features.clone()
    x_keys_clone = x_keys.clone()

    # Compute probabilities for the
    # duplicated batch
    with torch.no_grad():
        identity_probs = torch.softmax(
            model(x_features_clone, x_keys_clone),
            dim=TENSOR_CLASS_DIM,
        )

    # Calculate the maximum absolute difference
    # between the original and identity probabilities
    max_divergence = torch.max(
        torch.abs(original_probs - identity_probs),
    ).item()

    # Assert that the model is deterministic
    # and preserves identity
    assert (
        max_divergence < tests_config.model.behavioral.invariance.identity.tol
    )
