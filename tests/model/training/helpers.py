"""helpers.py

Module providing utility and setup functions necessary for running model
training tests.

These helpers handle the initialization of all components—data loaders,
model environment, optimizers, and configurations—required to execute various
functional and stability tests on the model's training process.

Functions:
    initialize_model_training_tests() -> tuple[
        DataLoader,
        Module,
        device,
        Module,
        Optimizer,
        PipelineConfig,
        TestsConfig,
    ]
        Initializes the entire environment (data, model, device, optimizer)
        needed for training tests.
    model_training_tests_setup() -> tuple[
        DataLoader,
        Module,
        device,
        Module,
        Optimizer,
        PipelineConfig,
        TestsConfig,
    ]
        Pytest fixture that provides the initialized environment for
        training tests.
    test_model_training_portability(model_training_tests_setup: tuple) -> None
        Tests if the model training process runs successfully on
        different supported hardware devices.
    test_model_training_end_to_end(model_training_tests_setup: tuple) -> None
        Tests the complete training pipeline execution and verifies
        the creation of the final model artifact.
    test_model_training_learnability(model_training_tests_setup: tuple) -> None
        Tests if the model can learn and reduce loss effectively
        when trained on a single batch of data.
    test_model_training_optimization_stability(model_training_tests_setup: tuple) -> None
        Tests the stability of the optimization process by asserting
        that the average loss decreases over subsequent training epochs.
    test_model_training_output_integrity(model_training_tests_setup: tuple) -> None
        Tests if the model's output tensor has the correct shape
        corresponding to the batch size and number of classes.
"""

import os

import numpy as np
import pytest
import torch
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from components.backpropagation.core.forward_runner import compute_forward
from components.data_loader.builder import build_data_loader
from components.data_loader.initializer import initialize_data_loader
from components.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.device.mover import move_to_device
from components.device.selector import select_device
from components.model.environment.initializer import (
    initialize_model_environment,
)
from components.model.io.locator import get_model_abs_path
from components.optimizer.builder import build_optimizer
from components.training.core.single_epoch_trainer import train_single_epoch
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from pipeline.const import DATASET_PROCESSED_TYPE
from pipeline.steps.trainer import train_model
from src.const import (
    DATASET_TRAINING_SPLIT_TYPE,
    RESOURCES_DEVICE_CPU_NAME,
    RESOURCES_DEVICE_CUDA_NAME,
    RESOURCES_DEVICE_NAMES,
)
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    MODEL_TRAINING_TESTS_NOT_SUCCESSFUL_TRAINING,
    MODEL_TRAINING_TESTS_SUCCESSFUL_TRAINING,
)


def initialize_model_training_tests() -> tuple[
    DataLoader,
    Module,
    device,
    Module,
    Optimizer,
    PipelineConfig,
    TestsConfig,
]:
    """Initializes the entire environment needed for running model
     training tests.

    This function loads the pipeline and test configurations, builds
    the training dataset and data loader, initializes the model environment
    (model, device, criterion), and prepares the optimizer.

    Returns:
        tuple[
            DataLoader,
            Module,
            device,
            Module,
            Optimizer,
            PipelineConfig,
            TestsConfig,
        ]:
            - training_loader: DataLoader for the training set.
            - model: The initialized model instance.
            - device: The device selected for training.
            - criterion: The loss function.
            - optimizer: The optimizer instance.
            - pipeline_config: The pipeline configuration object.
            - tests_config: The tests configuration object.
    """
    # Prepare pipeline configuration
    pipeline_config = prepare_pipeline_config()
    tests_config = prepare_tests_config()
    training_device = pipeline_config.resources.devices.training
    training_batch_size = pipeline_config.data_loader.training.batch_size
    training_shuffle = pipeline_config.data_loader.training.shuffle
    optimizer_type = pipeline_config.optimizer.type
    learning_rate = pipeline_config.optimizer.params.learning_rate
    weight_decay = pipeline_config.optimizer.params.weight_decay
    model_params = pipeline_config.model.params

    # Build training set
    _, training_loader = initialize_data_loader(
        DATASET_PROCESSED_TYPE,
        DATASET_TRAINING_SPLIT_TYPE,
        training_batch_size,
        training_shuffle,
        AccessLogsDataset,
        pipeline_config,
    )

    # Extract targets from training loader
    targets = extract_targets_from_data_loader(training_loader)

    # Prepare model environment
    device, criterion, model = initialize_model_environment(
        targets,
        training_device,
        pipeline_config,
        model_params=model_params,
    )

    # Build optimizer
    optimizer = build_optimizer(
        model,
        optimizer_type,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return (
        training_loader,
        model,
        device,
        criterion,
        optimizer,
        pipeline_config,
        tests_config,
    )


@pytest.fixture(scope="module")
def model_training_tests_setup() -> tuple[
    DataLoader,
    Module,
    device,
    Module,
    Optimizer,
    PipelineConfig,
    TestsConfig,
]:
    """Pytest fixture that initializes the entire environment
    for training tests.

    This fixture calls `initialize_model_training_tests()` and provides
    the resulting tuple of components to consuming test functions.

    Returns:
        tuple[
            DataLoader,
            Module,
            device,
            Module,
            Optimizer,
            PipelineConfig,
            TestsConfig,
        ]:
            - training_loader: DataLoader for the training set.
            - model: The initialized model instance.
            - device: The device selected for training.
            - criterion: The loss function.
            - optimizer: The optimizer instance.
            - pipeline_config: The pipeline configuration object.
            - tests_config: The tests configuration object.
    """
    return initialize_model_training_tests()


@pytest.mark.model_training_portability
def test_model_training_portability(model_training_tests_setup: tuple) -> None:
    """Tests if the model training process runs successfully on
    different supported hardware devices.

    This function iterates through all supported devices and attempts to run
    a single epoch of training after moving the model and criterion to the
    respective device. The test passes if no exceptions are raised during this process.

    Args:
        model_training_tests_setup (tuple): Fixture containing the initialized
                                            training environment.

    Raises:
        AssertionError: If an exception is caught during training on any device.
    """
    # Setup
    training_loader, model, _, criterion, optimizer, _, _ = (
        model_training_tests_setup
    )

    successful_training = MODEL_TRAINING_TESTS_SUCCESSFUL_TRAINING
    try:
        # Check train on different devices
        for device_name in RESOURCES_DEVICE_NAMES:
            # Check the current device name
            # availability before running training on,
            # selecting it if available
            # IMPORTANT: 'mps' is excluded as DDP is
            # not supported for it
            current_device = None
            if (  # noqa
                device_name == RESOURCES_DEVICE_CUDA_NAME
                and torch.cuda.is_available()
            ):
                current_device = select_device(device_name)
            elif (
                device_name == RESOURCES_DEVICE_CPU_NAME
                and torch.cpu.is_available()
            ):
                current_device = select_device(device_name)

            if current_device is not None:
                # Move everything to the selected device
                model = move_to_device(model, current_device)
                criterion = move_to_device(criterion, current_device)

                # Run training on the current device
                # (one epoch)
                train_single_epoch(
                    model,
                    training_loader,
                    optimizer,
                    criterion,
                    current_device,
                )
    except Exception:
        # Something went wrong, test failed
        successful_training = MODEL_TRAINING_TESTS_NOT_SUCCESSFUL_TRAINING

    # Check whether something went
    # wrong during training process
    assert successful_training


@pytest.mark.slow
@pytest.mark.model_training_end_to_end
def test_model_training_end_to_end(model_training_tests_setup: tuple) -> None:
    """Tests the complete end-to-end model training pipeline execution.

    This function runs the training model process and verifies that it
    completes without raising exceptions and that the final trained model
    artifact is successfully saved to the expected path.

    Args:
        model_training_tests_setup (tuple): Fixture containing the initialized
                                            training environment.

    Raises:
        AssertionError: If an exception occurs during training or if the final
                        model file is not found.
    """
    # Setup
    *_, pipeline_config, _ = model_training_tests_setup

    successful_training = MODEL_TRAINING_TESTS_SUCCESSFUL_TRAINING
    try:
        # Train the model (full)
        train_model()
    except Exception:
        # Something went wrong during
        # the end-to-end training process
        successful_training = MODEL_TRAINING_TESTS_NOT_SUCCESSFUL_TRAINING
    finally:
        model_path = get_model_abs_path(pipeline_config.data.general.mode)

        # Assert the training process went well and
        # it produced expected artifacts (trained model)
        assert successful_training
        assert os.path.exists(model_path)


@pytest.mark.model_training_learnability
def test_model_training_learnability(
    model_training_tests_setup: tuple,
) -> None:
    """Tests if the model can learn and reduce loss effectively when
    trained on a single batch of data.

    This function trains the model for the full number of configured epochs,
    but using a data loader restricted to only one batch. The function asserts that
    the final average loss is close to a predefined target loss within a given tolerance.

    Args:
        model_training_tests_setup (tuple): Fixture containing the initialized training
                                            environment.

    Raises:
        AssertionError: If the final average loss is not close to the target average loss.
    """
    # Setup
    (
        training_loader,
        model,
        device,
        criterion,
        optimizer,
        pipeline_config,
        tests_config,
    ) = model_training_tests_setup

    # Create data loaders for both training
    # and validation loaders with a single batch
    training_loader_one_batch = build_data_loader(
        Subset(
            training_loader.dataset,
            list(range(training_loader.batch_size)),
        ),
        training_loader.batch_size,
    )

    # Train the model on a single batch (full)
    avg_loss = None
    for epoch in range(pipeline_config.training.epochs):
        avg_loss = train_single_epoch(
            model,
            training_loader_one_batch,
            optimizer,
            criterion,
            device,
            epoch,
        )

    # Assert the average loss on training batches
    # is approximately equal to the expected one
    assert avg_loss == pytest.approx(
        tests_config.model.training.learnability.loss.target_avg,
        abs=tests_config.model.training.learnability.loss.tol,
    )


@pytest.mark.slow
@pytest.mark.model_training_optimization_stability
def test_model_training_optimization_stability(
    model_training_tests_setup: tuple,
) -> None:
    """Tests the stability of the optimization process across multiple epochs.

    This function trains the model for all configured epochs. After each epoch,
    new average loss is asserted to be strictly less than the average the loss
    from the previous epoch, indicating a stable reduction in the objective function.

    Args:
        model_training_tests_setup (tuple): Fixture containing the initialized
                                            training environment.

    Raises:
        AssertionError: If the average loss in a new epoch is not less than the
                        previous epoch's loss.
    """
    # Setup
    (
        training_loader,
        model,
        device,
        criterion,
        optimizer,
        pipeline_config,
        _,
    ) = model_training_tests_setup

    # Train model (full)
    previous_loss = np.inf
    for epoch in range(pipeline_config.training.epochs):
        # New epoch training
        avg_loss = train_single_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device,
            epoch,
        )

        # Assert the new average loss is
        # less than the previous one
        assert avg_loss < previous_loss


@pytest.mark.model_training_output_integrity
def test_model_training_output_integrity(
    model_training_tests_setup: tuple,
) -> None:
    """Tests if the model's output tensor has the correct shape.

    This functions takes a single batch of data, performs a forward pass
    through the model, and asserts that the output shape matches the expected
    dimensions.

    Args:
        model_training_tests_setup (tuple): Fixture containing the initialized
                                            training environment.

    Raises:
        AssertionError: If the output shape does not match the expected dimensions.
    """
    # Setup
    (training_loader, model, device, *_, pipeline_config, _) = (
        model_training_tests_setup
    )
    num_classes = (
        pipeline_config.data.general.keys.max
        - pipeline_config.data.general.keys.min
        + 1
    )

    # Get the first batch and unpack it
    batch = next(iter(training_loader))
    x_features, x_keys, y_key = batch

    # Calculate the outputs for the
    # current batch inputs
    _, outputs = compute_forward(batch, model, device)

    # Assert the model outputs have the expected shapes
    assert outputs.shape == torch.Size([len(x_features), num_classes])
