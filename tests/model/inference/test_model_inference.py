"""test_model_inference.py

Module containing the test function for verifying the performance and efficiency
of the model during the inference phase.

This module executes checks related to inference speed, ensuring that the model
meets performance requirements like acceptable average latency and minimum
required throughput.

Functions:
    test_model_inference() -> None
        Runs a suite of checks to validate the model's inference performance.
"""

import pytest

from tests.model.inference.helpers import (
    initialize_model_inference_tests,
    model_testing_tests_setup,  # noqa
    test_model_inference_avg_latency,
    test_model_inference_throughput,
)


@pytest.mark.model_inference
def test_model_inference() -> None:
    """Runs a suite of tests to validate the model's inference performance.

    This function initializes the inference environment and then executes
    tests focused on speed and efficiency:
        - Average Latency: Checks if the model's inference time (average)
                           is below a maximum allowed value.
        - Throughput: Checks if the model's processing rate (samples per second)
                      is above a minimum required value.

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    (testing_loader, model, device, tests_config) = (
        initialize_model_inference_tests()
    )

    # ----------------------------
    # Inference Tests
    # ----------------------------
    # Test the average latency of model
    # inference (i.e., whether it is below
    # a predefined value)
    test_model_inference_avg_latency(
        (testing_loader, model, device, tests_config),
    )

    # Test the model throughput (i.e.,
    # whether it is above a predefined value)
    test_model_inference_throughput(
        (testing_loader, model, device, tests_config),
    )


if __name__ == "__main__":
    test_model_inference()
