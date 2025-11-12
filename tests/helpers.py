"""helpers.py

Module providing utility functions to simplify the interaction with the
Deepchecks library, for setting up datasets and running test suites.

These functions standardize the conversion of pandas DataFrames into
Deepchecks Dataset objects and manage the execution, reporting, and
assertion logic for Deepchecks Suites across the tests.

Functions:
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
        html_as_widget: bool,
        html_requirejs: bool,
    ) -> None
        Executes a Deepchecks Suite on the provided dataset(s), saves the report,
        and asserts that all checks have passed.
"""

from pathlib import Path

from deepchecks import Dataset, Suite

import pandas as pd
from const import DATASET_COLUMN_REQUEST_NAME
from tests.const import (
    DEEP_CHECKS_SAVE_AS_HTML_AS_WIDGET,
    DEEP_CHECKS_SAVE_AS_HTML_REQUIREJS,
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
        html_as_widget (bool): Whether the HTML report should be displayed as a widget.
        html_requirejs (bool): Whether the HTML report should include JS code.

    Returns:
        None

    Raises:
        AssertionError: If `result.passed` is False, meaning one or more checks in the
                        suite failed their conditions.
    """
    # Run the suite
    result = suite.run(dc_training_set, dc_testing_set)

    # Save the results
    if Path(results_save_path).exists():
        Path(results_save_path).unlink()
    result.save_as_html(
        results_save_path,
        as_widget=html_as_widget,
        requirejs=html_requirejs,
    )

    assert result.passed
