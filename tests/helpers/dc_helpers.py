"""dc_helpers.py

Module providing utility functions to simplify the interaction with the
Deepchecks library, for setting up tests, datasets, and running test suites.

These functions standardize the conversion of pandas DataFrames into
Deepchecks Dataset objects and manage the setup, execution, reporting, and
assertion logic for Deepchecks Suites across the tests.

Functions:
    initialize_dc_tests(add_index_column: bool) -> tuple[
        Dataset,
        Dataset,
        Dataset,
        Module,
        TestsConfig
    ]
        Initialize all what is needed for running DeepChecks tests.
    create_dc_dataset(
        df: pd.DataFrame,
        target_column: str,
        cat_features: list[str],
        index_name: str
    ) -> Dataset
        Wraps a pandas DataFrame into a Deepchecks Dataset object.
    run_dc_suite(
        dc_training_set: Dataset,
        suite: Suite,
        results_save_path: str,
        dc_testing_set: Dataset,
        model: Any = None,
        html_as_widget: bool,
        html_requirejs: bool,
    ) -> None
        Executes a Deepchecks Suite on the provided dataset(s), saves the report,
        and asserts that all checks have passed.
"""

from pathlib import Path
from typing import Any

from deepchecks import Dataset, Suite

import pandas as pd
from torch.nn import Module

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.training_validation_splitter import (
    split_training_validation_sets,
)
from components.device.selector import select_device
from components.model.io.loader import load_model
from components.model.io.locator import get_model_abs_path
from const import (
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_TRAINING_SPLIT_TYPE,
    DATASET_TESTING_SPLIT_TYPE,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    DEEP_CHECKS_SAVE_AS_HTML_AS_WIDGET,
    DEEP_CHECKS_SAVE_AS_HTML_REQUIREJS,
    DATASET_COLUMN_TEMP_INDEX_NAME,
)


def initialize_dc_tests(
    add_index_column: bool,
) -> tuple[Dataset, Dataset, Dataset, Module, TestsConfig]:
    """Loads and prepares all what is needed for DeepChecks tests.

    This function orchestrates the setup of the DeepChecks testing environment.
    It begins by loading the global pipeline configuration alongside the specific
    tests' configuration. Next, it retrieves the processed training and testing
    datasets, and then splits the training data to create the validation subset
    using the configured ratio. All three resulting dataframes (training, validation,
    testing) are subsequently converted into Deepchecks Dataset objects. Additionally,
    a necessary temporary index column is added to each Deepchecks Dataset to facilitate
    index-based consistency checks. Finally, it loads the predictive model used for
    running the DeepChecks tests suite on.

    Args:
        add_index_column (bool): Whether to add the index column to the datasets or not.

    Returns:
        tuple[Dataset, Dataset, Dataset, PipelineConfig, Module, TestsConfig]:
            - dc_training_set (Dataset): Training data as a Deepchecks Dataset.
            - dc_validation_set (Dataset): Validation data as a Deepchecks Dataset.
            - dc_testing_set (Dataset): Testing data as a Deepchecks Dataset.
            - model (Module): The predictive model.
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
            len(training_set.data)
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
        training_set.data, index_name=DATASET_COLUMN_TEMP_INDEX_NAME
    )
    dc_validation_set = create_dc_dataset(
        validation_set.data, index_name=DATASET_COLUMN_TEMP_INDEX_NAME
    )
    dc_testing_set = create_dc_dataset(
        testing_set.data, index_name=DATASET_COLUMN_TEMP_INDEX_NAME
    )

    # Load the model
    model_path = get_model_abs_path(pipeline_config.data.general.mode)
    device = select_device(pipeline_config.resources.devices.testing)
    qengine = pipeline_config.model.optimizations.quantization.engine
    model = load_model(model_path, device, qengine)

    return (
        dc_training_set,
        dc_validation_set,
        dc_testing_set,
        model,
        tests_config,
    )


def create_dc_dataset(
    df: pd.DataFrame,
    target_column: str = DATASET_COLUMN_REQUEST_NAME,
    cat_features: list[str] = [],
    index_name: str = None,
) -> Dataset:
    """Wraps a pandas DataFrame into a Deepchecks Dataset object.

    This function standardizes the creation of a Deepchecks Dataset, which is
    the required input format for Deepchecks tests.

    Args:
        df (pd.DataFrame): The input pandas DataFrame containing the data.
        target_column (str): The name of the target/label column in the DataFrame.
        cat_features (list[str]): A list of column names to be treated as
                                  categorical features by Deepchecks.
        index_name (str): The name of the column to be used as the index
                          for the Deepchecks Dataset.

    Returns:
        Dataset: The initialized Deepchecks Dataset object.
    """
    return Dataset(
        df,
        label=target_column,
        cat_features=cat_features,
        index_name=index_name,
    )


def run_dc_suite(
    dc_training_set: Dataset,
    suite: Suite,
    results_save_path: str,
    dc_testing_set: Dataset = None,
    model: Any = None,
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
        model (Any): A predictive model to run the suite on.
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
        train_dataset=dc_training_set, test_dataset=dc_testing_set, model=model
    )

    # Save the results
    if Path(results_save_path).exists():
        Path(results_save_path).unlink()
    result.save_as_html(
        results_save_path,
        as_widget=html_as_widget,
        requirejs=html_requirejs,
    )

    assert result.passed
