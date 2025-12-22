"""helpers.py

Module containing utility functions and Pytest fixtures for setting up and executing
model inference tests.

These helpers handle the initialization of the best trained model, the testing data
loader, and the device environment, as well as assesses metrics like latency and throughput
meet expected requirements.

Functions:
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

from components.math.avg_calculator import calculate_average
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from tests.config.pydantic.tests_config import TestsConfig
from tests.model.helpers import initialize_inference_environment


@pytest.fixture(scope="module")
def model_testing_tests_setup() -> tuple[
    DataLoader,
    Module,
    device,
    PipelineConfig,
    TestsConfig,
]:
    """Pytest fixture that initializes the entire environment
    for inference tests.

    This fixture ensures that the setup (`initialize_inference_environment`)
    runs once per module, providing the necessary components to all inference
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


@pytest.mark.model_inference_avg_latency
@pytest.mark.testing
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
    (testing_loader, model, device, _, tests_config) = (
        model_testing_tests_setup
    )

    # Measure average latency over
    # all the batches inferred
    latencies = []
    with torch.no_grad():
        for batch in testing_loader:
            # Infer the current batch
            # keeping track of inference time
            start_time = time.perf_counter()
            x_features, x_keys, _ = batch
            model(x_features, x_keys)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

    # Compute average latency
    avg_latency = calculate_average(latencies)
    print(avg_latency)
    # Assert the average latency is close
    # enough to the expected one
    assert avg_latency == pytest.approx(
        tests_config.model.inference.latency.target_avg,
        abs=tests_config.model.inference.latency.tol,
    )


@pytest.mark.model_inference_throughput
@pytest.mark.testing
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
    testing_loader, model, device, _, tests_config = model_testing_tests_setup

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
            x_features, x_keys, _ = batch
            outputs = model(x_features, x_keys)
            tot_responses += len(outputs)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate throughput
    throughput = tot_responses / total_time

    # Assert throughput is above some minimum expected one
    assert throughput >= tests_config.model.inference.throughput.min
