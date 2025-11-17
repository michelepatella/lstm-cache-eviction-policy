"""helpers.py

Module containing utility functions and Pytest fixtures for setting up and executing
model inference tests.

These helpers handle the initialization of the best trained model, the testing data
loader, and the device environment, as well as assesses metrics like latency and throughput
meet expected requirements.

Functions:
    initialize_model_inference_tests() -> tuple[DataLoader, Module, device, TestsConfig]
        Initializes the environment for inference tests.
    model_testing_tests_setup() -> tuple[DataLoader, Module, device, TestsConfig]
        Pytest fixture providing the initialized environment.
    test_model_inference_avg_latency(model_testing_tests_setup: tuple) -> None
        Tests if the model's average inference latency is within the configured
        tolerance.
    test_model_inference_throughput(model_testing_tests_setup: tuple) -> None
        Tests if the model's throughput (samples per second) meets the minimum
        required value.
"""

import time

import pytest
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.backpropagation.core.forward_runner import compute_forward
from components.const import MODEL_EVAL_MODE
from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.device.mover import move_to_device
from components.math.avg_calculator import calculate_average
from components.model.best.initializer import initialize_best_model
from components.model.mode.setter import set_model_mode
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE
from src.const import DATASET_TESTING_SPLIT_TYPE
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig


def initialize_model_inference_tests() -> tuple[
    DataLoader,
    Module,
    device,
    TestsConfig,
]:
    """Initializes the entire environment needed for running model
    inference tests.

    This function loads the configuration, builds the testing data loader,
    and initializes the trained best model, moving it to the specified
    testing device.

    Returns:
        tuple[
            DataLoader,
            Module,
            device,
            TestsConfig,
        ]:
            - testing_loader: DataLoader for the testing set.
            - model: The loaded, best-performing model instance.
            - device: The device selected for inference.
            - tests_config: The validated tests configuration object.
    """
    # Prepare configuration
    pipeline_config = prepare_pipeline_config()
    tests_config = prepare_tests_config()
    data_mode = pipeline_config.data.general.mode
    testing_batch_size = pipeline_config.data_loader.testing.batch_size
    testing_shuffle = pipeline_config.data_loader.testing.shuffle
    testing_device = pipeline_config.resources.devices.testing
    qengine = pipeline_config.model.optimizations.quantization.engine

    # Build testing loader
    _, testing_loader = initialize_data_loader(
        DATASET_PROCESSED_TYPE,
        DATASET_TESTING_SPLIT_TYPE,
        testing_batch_size,
        testing_shuffle,
        AccessLogsDataset,
        pipeline_config,
    )

    # Trained model setup for testing
    device, _, model = initialize_best_model(
        data_mode,
        testing_device,
        pipeline_config,
        testing_loader,
        qengine=qengine,
    )

    return (testing_loader, model, device, tests_config)


@pytest.fixture(scope="module")
def model_testing_tests_setup() -> tuple[
    DataLoader,
    Module,
    device,
    TestsConfig,
]:
    """Pytest fixture that initializes the entire environment
    for inference tests.

    This fixture ensures that the setup (`initialize_model_inference_tests`)
    runs once per module, providing the necessary components to all inference
    test functions.
    """
    return initialize_model_inference_tests()


@pytest.mark.model_inference_avg_latency
def test_model_inference_avg_latency(model_testing_tests_setup: tuple) -> None:
    """Tests if the model's average inference latency is within the
    configured tolerance.

    The test measures the time taken for inference on every batch in the
    testing loader, calculates the average latency, and asserts that this
    average is approximately equal to the target latency defined in the
    test configuration, considering the specified absolute tolerance.

    Args:
        model_testing_tests_setup (tuple): Fixture containing the initialized
                                           inference environment.

    Raises:
        AssertionError: If the computed average latency exceeds the expected
                        value plus tolerance.
    """
    # Setup
    (testing_loader, model, device, tests_config) = model_testing_tests_setup

    # Move model to device and set evaluation mode
    model = move_to_device(model, device)
    set_model_mode(model, MODEL_EVAL_MODE)

    # Measure average latency over
    # all the batches inferred
    latencies = []
    with torch.no_grad():
        for batch in testing_loader:
            # Infer the current batch
            # keeping track of inference time
            start_time = time.perf_counter()
            compute_forward(batch, model, device)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

    # Compute average latency
    avg_latency = calculate_average(latencies)

    # Assert the average latency is close
    # enough to the expected one
    assert avg_latency == pytest.approx(
        tests_config.model.inference.latency.target_avg,
        abs=tests_config.model.inference.latency.tol,
    )


@pytest.mark.model_inference_throughput
def test_model_inference_throughput(model_testing_tests_setup: tuple) -> None:
    """Tests if the model's throughput meets the minimum required value.

    Throughput is calculated as the total number of model responses served
    over the total time required to process all batches in the testing loader.
    The result must be greater than or equal to the minimum throughput value
    defined in the test configuration.

    Args:
        model_testing_tests_setup (tuple): Fixture containing the initialized
                                           inference environment.

    Raises:
        AssertionError: If the computed throughput is less than the minimum
                        required value.
    """
    # Setup
    testing_loader, model, device, tests_config = model_testing_tests_setup

    # Move model to device and set evaluation mode
    model = move_to_device(model, device)
    set_model_mode(model, MODEL_EVAL_MODE)

    # Infer all the batches while keeping
    # track of the number of model outputs
    # and total time required to serve responses
    tot_responses = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in testing_loader:
            # Infer the current batch and
            # keep track of the number of
            # model responses
            _, outputs = compute_forward(batch, model, device)
            tot_responses += len(outputs)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate throughput
    throughput = tot_responses / total_time

    # Assert throughput is above some minimum expected one
    assert throughput >= tests_config.model.inference.throughput.min
