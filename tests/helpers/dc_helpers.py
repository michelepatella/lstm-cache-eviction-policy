"""dc_helpers.py

Module providing utility functions to simplify the interaction with the
Deepchecks library, for setting up tests, datasets, and running test suites.

These functions standardize the conversion of pandas DataFrames into
Deepchecks Dataset objects and manage the setup, execution, reporting, and
assertion logic for Deepchecks Suites across the tests.

Functions:
    initialize_dc_tests(add_index_column: bool, remove_seq_len: bool) -> tuple[
        AccessLogsDataset,
        AccessLogsDataset,
        AccessLogsDataset,
        Dataset,
        Dataset,
        Dataset,
        PipelineConfig,
        TestsConfig,
    ]
        Initialize all what is needed for running DeepChecks tests.
    create_dc_dataset(
        df: pd.DataFrame,
        seq_len: int = None,
        target_column: str = DATASET_COLUMN_REQUEST_NAME,
        cat_features: list[str] = [],
        index_name: str = None,
        label_type: str = DEEP_CHECKS_LABEL_TYPE,
    ) -> Dataset
        Wraps a pandas DataFrame into a Deepchecks Dataset object.
    compute_dc_model_predictions(
        training_set: AccessLogsDataset,
        testing_set: AccessLogsDataset,
        pipeline_config: PipelineConfig,
        training_loader: DataLoader = None,
        testing_loader: DataLoader = None,
        model: Module = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]
        Initializes the model and computes class predictions and probabilities
        for both training and testing sets.
    run_dc_suite(
        dc_training_set: Dataset,
        suite: Suite,
        results_save_path: str,
        dc_testing_set: Dataset = None,
        y_pred_train: np.ndarray = None,
        y_pred_test: np.ndarray = None,
        y_proba_train: np.ndarray = None,
        y_proba_test: np.ndarray = None,
        html_as_widget: bool = DEEP_CHECKS_SAVE_AS_HTML_AS_WIDGET,
        html_requirejs: bool = DEEP_CHECKS_SAVE_AS_HTML_REQUIREJS,
    ) -> None
        Executes a Deepchecks Suite on the provided dataset(s), saves the report,
        and asserts that all checks have passed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from deepchecks import Dataset, Suite
from numpy import ndarray
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import TENSOR_BATCH_DIM, TENSOR_CLASS_DIM
from components.data_loader.builder import build_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.training_validation_splitter import (
    split_training_validation_sets,
)
from components.evaluation.model.evaluator import evaluate_model
from components.model.best.initializer import initialize_best_model
from const import (
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_TESTING_SPLIT_TYPE,
    DATASET_TRAINING_SPLIT_TYPE,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from pipeline.const import DATASET_PROCESSED_TYPE
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    DATASET_COLUMN_TEMP_INDEX_NAME,
    DEEP_CHECKS_LABEL_TYPE,
    DEEP_CHECKS_SAVE_AS_HTML_AS_WIDGET,
    DEEP_CHECKS_SAVE_AS_HTML_REQUIREJS,
)


def initialize_dc_tests(
    add_index_column: bool,
    remove_seq_len: bool,
) -> tuple[
    AccessLogsDataset,
    AccessLogsDataset,
    AccessLogsDataset,
    Dataset,
    Dataset,
    Dataset,
    PipelineConfig,
    TestsConfig,
]:
    """Loads and prepares all what is needed for DeepChecks tests.

    This function orchestrates the setup of the DeepChecks testing environment.
    It begins by loading the global pipeline configuration alongside the specific
    tests' configuration. Next, it retrieves the processed training and testing
    datasets, and then splits the training data to create the validation subset using
    the configured ratio. All three resulting dataframes (training, validation, testing)
    are subsequently converted into Deepchecks Dataset objects. Additionally, a necessary
    temporary index column is added to each Deepchecks Dataset to facilitate index-based
    consistency checks.

    Args:
        add_index_column (bool): Whether to add the index column to the datasets or not.
        remove_seq_len (bool): Whether to remove sequence length from data or not.

    Returns:
        tuple[
            AccessLogsDataset,
            AccessLogsDataset,
            AccessLogsDataset,
            Dataset,
            Dataset,
            Dataset,
            PipelineConfig,
            TestsConfig,
        ]:
            - training_set (AccessLogsDataset): The training dataset.
            - testing_set (AccessLogsDataset): The testing dataset.
            - validation_set (AccessLogsDataset): The validation dataset.
            - dc_training_set (Dataset): Training data as a Deepchecks Dataset.
            - dc_validation_set (Dataset): Validation data as a Deepchecks Dataset.
            - dc_testing_set (Dataset): Testing data as a Deepchecks Dataset.
            - pipeline_config (PipelineConfig): The validated pipeline configuration
                                                object.
            - tests_config (TestsConfig): The validated tests configuration object.
    """
    # Prepare configurations
    pipeline_config = prepare_pipeline_config()
    tests_config = prepare_tests_config()

    # Load training, validation, and testing sets
    training_set, validation_set = split_training_validation_sets(
        AccessLogsDataset(
            DATASET_PROCESSED_TYPE,
            DATASET_TRAINING_SPLIT_TYPE,
            pipeline_config,
        ),
        pipeline_config.dataset.splits.validation,
    )
    testing_set = AccessLogsDataset(
        DATASET_PROCESSED_TYPE,
        DATASET_TESTING_SPLIT_TYPE,
        pipeline_config,
    )

    # Add a temporary index column
    # to all the datasets if requested
    if add_index_column:
        training_set.data[DATASET_COLUMN_TEMP_INDEX_NAME] = range(
            len(training_set.data),
        )
        validation_set.data[DATASET_COLUMN_TEMP_INDEX_NAME] = range(
            len(training_set.data),
            len(training_set.data) + len(validation_set.data),
        )
        testing_set.data[DATASET_COLUMN_TEMP_INDEX_NAME] = range(
            len(training_set.data) + len(validation_set.data),
            len(training_set.data)
            + len(validation_set.data)
            + len(testing_set.data),
        )

    # Convert datasets to DeepChecks Dataset objects
    dc_training_set = create_dc_dataset(
        training_set.data,
        seq_len=pipeline_config.model.sequence.length
        if remove_seq_len
        else None,
        index_name=DATASET_COLUMN_TEMP_INDEX_NAME
        if add_index_column
        else None,
    )
    dc_validation_set = create_dc_dataset(
        validation_set.data,
        seq_len=pipeline_config.model.sequence.length
        if remove_seq_len
        else None,
        index_name=DATASET_COLUMN_TEMP_INDEX_NAME
        if add_index_column
        else None,
    )
    dc_testing_set = create_dc_dataset(
        testing_set.data,
        seq_len=pipeline_config.model.sequence.length
        if remove_seq_len
        else None,
        index_name=DATASET_COLUMN_TEMP_INDEX_NAME
        if add_index_column
        else None,
    )

    return (
        training_set,
        validation_set,
        testing_set,
        dc_training_set,
        dc_validation_set,
        dc_testing_set,
        pipeline_config,
        tests_config,
    )


def create_dc_dataset(
    df: pd.DataFrame,
    seq_len: int = None,
    target_column: str = DATASET_COLUMN_REQUEST_NAME,
    cat_features: list[str] = [],
    index_name: str = None,
    label_type: str = DEEP_CHECKS_LABEL_TYPE,
) -> Dataset:
    """Wraps a pandas DataFrame into a Deepchecks Dataset object.

    This function standardizes the creation of a Deepchecks Dataset, which is
    the required input format for Deepchecks tests.

    Args:
        df (pd.DataFrame): The input pandas DataFrame containing the data.
        seq_len (int): Data sequence length for the predictive model.
        target_column (str): The name of the target/label column in the DataFrame.
        cat_features (list[str]): A list of column names to be treated as
                                  categorical features by Deepchecks.
        index_name (str): The name of the column to be used as the index
                          for the Deepchecks Dataset.
        label_type (str): The type of task to solve.

    Returns:
        Dataset: The initialized Deepchecks Dataset object.
    """
    return Dataset(
        df.iloc[seq_len:] if seq_len is not None else df,
        label=target_column,
        cat_features=cat_features,
        index_name=index_name,
        label_type=label_type,
    )


def compute_dc_model_predictions(
    training_set: AccessLogsDataset,
    testing_set: AccessLogsDataset,
    pipeline_config: PipelineConfig,
    training_loader: DataLoader = None,
    testing_loader: DataLoader = None,
    model: Module = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Initializes the best model and computes predictions/probabilities
     for DeepChecks.

    This function sets up the data loaders and initializes the trained model
    environment for both the training and testing sets. It then evaluates the
    model on both sets to obtain the raw logits, which are converted into
    class predictions and probabilities (using softmax). These results are
    returned as numpy arrays, ready to be consumed by Deepchecks for model-related
    tests.

    Args:
        training_set (AccessLogsDataset): The training dataset.
        testing_set (AccessLogsDataset): The testing dataset.
        pipeline_config (PipelineConfig): The pipeline configuration object.
        training_loader (DataLoader): The training data loader.
        testing_loader (DataLoader): The testing data loader.
        model (Module): The PyTorch model to run inference on.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray]:
            - y_pred_train (np.ndarray): Predicted classes for the training set.
            - y_pred_test (np.ndarray): Predicted classes for the testing set.
            - y_proba_train (np.ndarray): Predicted probabilities for the training set.
            - y_proba_test (np.ndarray): Predicted probabilities for the testing set.
    """
    # Prepare configuration
    data_mode = pipeline_config.data.general.mode
    training_batch_size = pipeline_config.data_loader.training.batch_size
    testing_batch_size = pipeline_config.data_loader.testing.batch_size
    training_device_type = pipeline_config.resources.devices.training
    testing_device_type = pipeline_config.resources.devices.testing
    qengine = pipeline_config.model.optimizations.quantization.engine
    num_classes = (
        pipeline_config.data.general.keys.max
        - pipeline_config.data.general.keys.min
        + 1
    )
    num_workers = max(
        pipeline_config.resources.general.num_cpus,
        pipeline_config.resources.general.num_gpus,
    )

    # Prepare data loaders
    if training_loader is None:
        training_loader = build_data_loader(
            training_set,
            training_batch_size,
        )
    if testing_loader is None:
        testing_loader = build_data_loader(
            testing_set,
            testing_batch_size,
        )

    # Initialize best model environment
    # both for training and testing sets
    training_device, training_criterion, initialized_model = (
        initialize_best_model(
            data_mode,
            training_device_type,
            pipeline_config,
            training_loader,
            qengine=qengine,
        )
    )
    testing_device, testing_criterion, _ = initialize_best_model(
        data_mode,
        testing_device_type,
        pipeline_config,
        testing_loader,
        qengine=qengine,
    )

    if model is None:
        model = initialized_model

    # Evaluate model both on training and
    # testing sets
    (_, _, train_outputs, y_pred_train, *_) = evaluate_model(
        model,
        training_loader,
        training_criterion,
        training_device,
        num_workers,
    )
    (_, _, test_outputs, y_pred_test, *_) = evaluate_model(
        model,
        testing_loader,
        testing_criterion,
        testing_device,
        num_workers,
    )

    # Extract probabilities from raw logits
    train_outputs = torch.cat(train_outputs, dim=TENSOR_BATCH_DIM)
    test_outputs = torch.cat(test_outputs, dim=TENSOR_BATCH_DIM)
    y_proba_train = (
        torch.softmax(train_outputs, dim=TENSOR_CLASS_DIM)
        .detach()
        .cpu()
        .numpy()
    )
    y_proba_test = (
        torch.softmax(test_outputs, dim=TENSOR_CLASS_DIM)
        .detach()
        .cpu()
        .numpy()
    )

    # Reshape np.ndarray
    num_samples_train = y_proba_train.size // num_classes
    y_proba_train = y_proba_train.reshape(num_samples_train, num_classes)
    num_samples_test = y_proba_test.size // num_classes
    y_proba_test = y_proba_test.reshape(num_samples_test, num_classes)

    return y_pred_train, y_pred_test, y_proba_train, y_proba_test


def run_dc_suite(
    dc_training_set: Dataset,
    suite: Suite,
    results_save_path: str,
    dc_testing_set: Dataset = None,
    y_pred_train: np.ndarray = None,
    y_pred_test: np.ndarray = None,
    y_proba_train: np.ndarray = None,
    y_proba_test: np.ndarray = None,
    html_as_widget: bool = DEEP_CHECKS_SAVE_AS_HTML_AS_WIDGET,
    html_requirejs: bool = DEEP_CHECKS_SAVE_AS_HTML_REQUIREJS,
) -> None:
    """Executes the Deepchecks test suite, saves the results report,
     and asserts passage.

    This function runs all checks contained within the provided Deepchecks
    Suite on the target datasets, displays a summary of the results, saves a
    detailed HTML report, and raises an assertion error if any check fails
    its defined condition.

    Args:
        dc_training_set (Dataset): The Deepchecks training Dataset object to be tested.
        suite (Suite): The configured Deepchecks Suite containing integrity checks.
        results_save_path (str): The file path where the HTML report of the results
                                 should be saved.
        dc_testing_set (Dataset): The Deepchecks testing Dataset object to be tested.
        y_pred_train (np.ndarray): Classes predicted by the model on training set.
        y_pred_test (np.ndarray): Classes predicted by the model on testing set.
        y_proba_train (np.ndarray): Probabilities predicted by the model on training set.
        y_proba_test (np.ndarray): Probabilities predicted by the model on testing set.
        html_as_widget (bool): Whether the HTML report should be displayed as a widget.
        html_requirejs (bool): Whether the HTML report should include JS code.

    Returns:
        None

    Raises:
        AssertionError: If `result.passed` is False, meaning one or more checks in the
                        suite failed their conditions.
    """
    # Run the suite
    result = suite.run(
        train_dataset=dc_training_set,
        test_dataset=dc_testing_set,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        y_proba_train=y_proba_train,
        y_proba_test=y_proba_test,
    )

    # Save the results (create the
    # directory if it does not exist yet)
    results_path = Path(results_save_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()
    result.save_as_html(
        str(results_path),
        as_widget=html_as_widget,
        requirejs=html_requirejs,
    )

    assert result.passed
