"""helpers.py

Module containing tests designed to verify the model's minimum functional ability
to handle and predict specific, known input sequences.

Functions:
    model_minimum_functionality_tests_setup() -> tuple[
        DataLoader,
        Module,
        device,
        PipelineConfig,
        TestsConfig,
    ]
        Pytest fixture initializing the inference environment.
    test_model_minimum_functionality_on_sequence(
        model_minimum_functionality_tests_setup: tuple,
        features_sequence: list[list[float]],
        keys_sequence: list[int],
        expected_key: int,
    ) -> None
        Tests the model's prediction for a single, specific input sequence.
"""

import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import LIST_FIRST_IDX, TENSOR_CLASS_DIM
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from tests.config.pydantic.tests_config import TestsConfig
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
    runs once per module, providing the necessary components to all minimum
    functionality test functions.

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


@pytest.mark.model_behavioral_minimum_functionality_on_sequence
def test_model_minimum_functionality_on_sequence(
    model_minimum_functionality_tests_setup: tuple,
    features_sequence: list[list[float]],
    keys_sequence: list[int],
    expected_key: int,
) -> None:
    """Tests the model's prediction on a single, known input sequence.

    This test verifies the model's basic functional correctness by asserting
    that when presented with a specific sequence of features and keys, it
    outputs the predefined expected next key. The test assumes external
    parametrization provides the input sequences and the expected key.

    Args:
        model_minimum_functionality_tests_setup (tuple): Fixture containing
                                                         the model and inference
                                                         environment.
        features_sequence (list[list[float]]): The input feature sequence to feed
                                               to the model.
        keys_sequence (list[int]): The input key sequence (requests) to feed
                                   to the model.
        expected_key (int): The known, correct key the model must predict.

    Raises:
        AssertionError: If the model's top predicted key does not match the
                        expected one.
    """
    # Setup
    (
        _,
        model,
        device,
        _,
        _,
    ) = model_minimum_functionality_tests_setup

    # Convert inputs to tensors
    x_features = torch.tensor(
        [features_sequence],
        dtype=torch.float,
        device=device,
    )
    x_keys = torch.tensor([keys_sequence], dtype=torch.long, device=device)

    # Get the predicted class by making inference
    with torch.no_grad():
        predicted_class = torch.argmax(
            torch.softmax(model(x_features, x_keys), dim=TENSOR_CLASS_DIM),
            dim=TENSOR_CLASS_DIM,
        )

    # Extract the prediction
    prediction = predicted_class[LIST_FIRST_IDX].item()

    # Assert that the model prediction is
    # equal to the expected one
    assert prediction == expected_key
