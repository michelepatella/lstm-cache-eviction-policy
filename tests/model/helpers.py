"""helpers.py

Module providing utility functions for model tests.

This module provided utilities for model tests, including the initialization of
inference environment.

Functions:
    initialize_inference_environment() -> tuple[
        DataLoader, Module, device, PipelineConfig, TestsConfig,
    ]
        Initializes the complete environment required for running model tests.
"""

from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import MODEL_EVAL_MODE
from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.model.best.initializer import initialize_best_model
from components.model.mode.setter import set_model_mode
from const import DATASET_TESTING_SPLIT_TYPE
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from pipeline.const import DATASET_PROCESSED_TYPE
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig


def initialize_inference_environment() -> tuple[
    DataLoader,
    Module,
    device,
    PipelineConfig,
    TestsConfig,
]:
    """Initializes the entire environment needed for running model
    inference functionalities.

    This function loads the configuration, builds the testing data loader,
    and initializes the trained best model, setting it to the evaluation mode.

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

    # Trained model setup
    device, _, model = initialize_best_model(
        data_mode,
        testing_device,
        pipeline_config,
        testing_loader,
        qengine=qengine,
    )

    # Set model to evaluation mode
    set_model_mode(model, MODEL_EVAL_MODE)

    return (testing_loader, model, device, pipeline_config, tests_config)
