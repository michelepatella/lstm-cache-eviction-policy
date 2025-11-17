"""test_model_training.py

Module containing the test function for verifying the integrity,
stability, and correctness of the model training process.

This module aggregates several tests that validate key aspects
of the training pipeline, ensuring the model is learnable, the process
is stable, and outputs are correctly shaped.
Functions:
    test_model_training() -> None
        Runs a suite of checks to validate the model training process.
"""

import pytest

from tests.model.training.helpers import (
    initialize_model_training_tests,
    model_training_tests_setup,  # noqa
    test_model_training_end_to_end,
    test_model_training_learnability,
    test_model_training_optimization_stability,
    test_model_training_output_integrity,
    test_model_training_portability,
)


@pytest.mark.model.training
def test_model_training() -> None:
    """Runs a suite of tests to validate the model training process.

    This function executes a series of functional and stability checks
    on the training process, including:
        - Portability: Ensures training works across different devices.
        - End-to-End: Verifies that the training process completes without
                      errors.
        - Learnability: Confirms the model can achieve good performance over a
                        single batch.
        - Optimization Stability: Checks that the average training loss consistently
                                  decreases.
        - Output Integrity: Ensures the model outputs have the correct shape and
                            structure.

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    (
        training_loader,
        model,
        device,
        criterion,
        optimizer,
        pipeline_config,
        tests_config,
    ) = initialize_model_training_tests()

    # ----------------------------
    # Training Tests
    # ----------------------------
    # Test model training portability
    # (i.e., training works well on different
    # devices)
    test_model_training_portability(
        (
            training_loader,
            model,
            device,
            criterion,
            optimizer,
            pipeline_config,
            tests_config,
        ),
    )

    # Test model training end-to-end
    # (i.e., the training process produces the
    # expected results, so nothing goes wrong)
    test_model_training_end_to_end(
        (
            training_loader,
            model,
            device,
            criterion,
            optimizer,
            pipeline_config,
            tests_config,
        ),
    )

    # Test model training learnability
    # (i.e., the model achieves good performance
    # over a single batch)
    test_model_training_learnability(
        (
            training_loader,
            model,
            device,
            criterion,
            optimizer,
            pipeline_config,
            tests_config,
        ),
    )

    # Test model training optimization stability
    # (i.e., the average loss on the training
    # set keep decreasing over time)
    test_model_training_optimization_stability(
        (
            training_loader,
            model,
            device,
            criterion,
            optimizer,
            pipeline_config,
            tests_config,
        ),
    )

    # Test model training output integrity
    # (i.e., the outputs of the model have
    # the expected shape)
    test_model_training_output_integrity(
        (
            training_loader,
            model,
            device,
            criterion,
            optimizer,
            pipeline_config,
            tests_config,
        ),
    )


if __name__ == "__main__":
    test_model_training()
