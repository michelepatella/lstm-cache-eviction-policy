"""helpers.py

Module dedicated to preparing the necessary data and configuration structure
for executing data integrity tests.

This module provides utility functions for initializing the testing environment
(loading data into Deepchecks Dataset and parsing configurations) and for
executing and reporting the results of the Deepchecks integrity tests suite.

Functions:
    initialize_data_integrity_tests(
        dataset_type: str,
        data_mode: str
    ) -> tuple[Dataset, TestsConfig]
        Loads the specified dataset and the tests configuration, returning both
        as Deepchecks Dataset and Pydantic TestsConfig objects.
    run_data_integrity_tests(
        dc_dataset: Dataset,
        suite: Suite,
        results_save_path: str
    ) -> None
        Executes a Deepchecks Suite on the provided dataset, saves the report,
        and asserts that all checks have passed.
"""

from pathlib import Path

from deepchecks import Dataset, Suite

from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from const import DATASET_COLUMN_REQUEST_NAME
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import (
    DATA_INTEGRITY_RESULTS_HTML_AS_WIDGET,
    DATA_INTEGRITY_RESULTS_HTML_REQUIREJS,
)


def initialize_data_integrity_tests(
    dataset_type: str,
    data_mode: str,
) -> tuple[Dataset, TestsConfig]:
    """Loads the dataset and prepares the configuration for data
    integrity tests.

    This function loads the specified pandas DataFrame — converting it
    into a Deepchecks Dataset object — and validates the tests configuration
    loaded from the YAML file.

    Args:
        dataset_type (str): The type of the dataset.
        data_mode (str): The mode of the data.

    Returns:
        tuple[Dataset, TestsConfig]:
            - dc_dataset: The loaded data wrapped as a Deepchecks Dataset object.
            - tests_config: The validated tests configuration object.
    """
    # Load the dataset and wrap into
    # a Deepchecks Dataset object.
    df = load_dataset(get_dataset_abs_path(dataset_type, data_mode))
    dc_dataset = Dataset(
        df,
        label=DATASET_COLUMN_REQUEST_NAME,
        cat_features=[],
    )

    # Prepare the tests configuration
    tests_config = prepare_tests_config()

    return dc_dataset, tests_config


def run_data_integrity_tests(
    dc_dataset: Dataset,
    suite: Suite,
    results_save_path: str,
) -> None:
    """Executes the Deepchecks test suite, saves the results report,
     and asserts passage.

    This function runs all checks contained within the provided Deepchecks
    Suite on the target dataset, displays a summary of the results, saves a
    detailed HTML report, and raises an assertion error if any check fails
    its defined condition.

    Args:
        dc_dataset (Dataset): The Deepchecks Dataset object to be tested.
        suite (Suite): The configured Deepchecks Suite containing integrity checks.
        results_save_path (str): The file path where the HTML report of the results
                                 should be saved.

    Returns:
        None

    Raises:
        AssertionError: If `result.passed` is False, meaning one or more checks in the
                        suite failed their conditions.
    """
    # Run the suite
    result = suite.run(dc_dataset)

    # Save the results
    if Path(results_save_path).exists():
        Path(results_save_path).unlink()
    result.save_as_html(
        results_save_path,
        as_widget=DATA_INTEGRITY_RESULTS_HTML_AS_WIDGET,
        requirejs=DATA_INTEGRITY_RESULTS_HTML_REQUIREJS,
    )

    assert result.passed
