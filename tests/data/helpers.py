"""helpers.py

Comprehensive module dedicated to the setup, definition, and execution of data
quality validation using the Great Expectations (GX) framework.

This manager module centralizes all necessary components for robust data testing:
    - Initialization: Functions for loading the dataset and configuring the GX
                      environment (Context, Datasource, Asset, Batch, Suite).
    - Expectation Definition: A collection of utility functions to programmatically
                              add various data quality expectations to an
                              ExpectationSuite, covering structural checks
                              (existence, count, order), content checks
                              (not null, type, range), and integrity checks
                              (row count).
    - Execution: Logic to define the Validation Definition and Checkpoint, run
                 the tests against the DataFrame, and assert the successful
                 completion of all defined expectations.

Functions:
    initialize_dataset_testing(
        dataset_type: str,
        data_mode: str,
        data_source_name: str,
        dataframe_asset_name: str,
        batch_definition_name: str,
        expectation_suite_name: str,
    ) -> tuple[
        pd.DataFrame,
        EphemeralDataContext,
        PandasDatasource,
        DataFrameAsset,
        BatchDefinition,
        ExpectationSuite,
    ]
        Sets up the required Great Expectations environment.

    add_column_not_null_expectations(
        suite: ExpectationSuite,
        columns: list[str],
    ) -> None
        Adds non-null constraints to specified columns.

    add_column_range_expectations(
        suite: ExpectationSuite,
        column_ranges: dict[str, tuple[float, float]],
    ) -> None
        Adds range constraints to specified columns.

    add_column_existence_expectations(
        suite: ExpectationSuite,
        column_indexes: dict[str, int],
    ) -> None
        Checks for column presence and ordinal index.

    add_column_type_expectations(
        suite: ExpectationSuite,
        column_types: dict[str, str],
    ) -> None
        Checks for expected data types in columns.

    add_column_count_expectation(
        suite: ExpectationSuite,
        column_count: int,
    ) -> None
        Checks for the total number of columns.

    add_column_order_expectation(
        suite: ExpectationSuite,
        columns: list[str],
    ) -> None
        Checks for the precise order of columns.

    add_row_count_expectation(
        suite: ExpectationSuite,
        row_count: int,
    ) -> None
        Checks for the expected total number of rows.

    run_dataset_testing(
        context: EphemeralDataContext,
        batch_definition: BatchDefinition,
        suite: ExpectationSuite,
        df: pd.DataFrame,
        validation_definition_name: str,
        checkpoint_name: str,
    ) -> None
        Executes the defined validation Checkpoint and asserts success.
"""

import great_expectations as gx
import pandas as pd
from great_expectations import ExpectationSuite
from great_expectations.core.batch_definition import BatchDefinition
from great_expectations.data_context import EphemeralDataContext
from great_expectations.datasource.fluent import PandasDatasource
from great_expectations.datasource.fluent.pandas_datasource import (
    DataFrameAsset,
)

from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from tests.const import (
    DATA_TESTING_BATCH_DEFINITION_NAME_DEFAULT,
    DATA_TESTING_CHECKPOINT_BATCH_PARAMETERS_DATAFRAME_KEY_NAME,
    DATA_TESTING_CHECKPOINT_NAME,
    DATA_TESTING_DATA_SOURCE_NAME_DEFAULT,
    DATA_TESTING_DATAFRAME_ASSET_NAME_DEFAULT,
    DATA_TESTING_EXPECTATION_SUITE_NAME_DEFAULT,
    DATA_TESTING_VALIDATION_DEFINITION_NAME,
)


def initialize_dataset_testing(
    dataset_type: str,
    data_mode: str,
    data_source_name: str = DATA_TESTING_DATA_SOURCE_NAME_DEFAULT,
    dataframe_asset_name: str = DATA_TESTING_DATAFRAME_ASSET_NAME_DEFAULT,
    batch_definition_name: str = DATA_TESTING_BATCH_DEFINITION_NAME_DEFAULT,
    expectation_suite_name: str = DATA_TESTING_EXPECTATION_SUITE_NAME_DEFAULT,
) -> tuple[
    pd.DataFrame,
    EphemeralDataContext,
    PandasDatasource,
    DataFrameAsset,
    BatchDefinition,
    ExpectationSuite,
]:
    """Initializes the Great Expectations testing environment for a dataset.

    This function loads a specified dataset into a Pandas DataFrame and
    configures all necessary Great Expectations components, including an
    Ephemeral Data Context, a Pandas Datasource, a DataFrame Asset, a
    Batch Definition, and an Expectation Suite.

    Args:
        dataset_type (str): The type of dataset to load.
        data_mode (str): The processing mode of the data.
        data_source_name (str): Name assigned to the Pandas Datasource.
        dataframe_asset_name (str): Name assigned to the DataFrame Asset.
        batch_definition_name (str): Name assigned to the Batch Definition.
        expectation_suite_name (str): Base name for the Expectation Suite.

    Returns:
        tuple[
            pd.DataFrame,
            EphemeralDataContext,
            PandasDatasource,
            DataFrameAsset,
            BatchDefinition,
            ExpectationSuite
        ]:
            - df: The dataset loaded into a Pandas DataFrame.
            - context: The temporary Great Expectations Data Context.
            - data_source: The configured Great Expectations Datasource.
            - data_asset: The Great Expectations Data Asset wrapper for
                          the DataFrame.
            - batch_definition: The definition used to select the entire
                                DataFrame for validation.
            - suite: The initialized Great Expectations suite where expectations
                     will be added.
    """
    # Load dataset
    df = load_dataset(get_dataset_abs_path(dataset_type, data_mode))

    # Create a data context
    context = gx.get_context()

    # Define a datasource
    data_source = context.data_sources.add_pandas(name=data_source_name)

    # Create a data asset for the dataframe
    data_asset = data_source.add_dataframe_asset(name=dataframe_asset_name)

    # Create a batch definition and a batch to validate
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        batch_definition_name,
    )

    # Create a suite of expectations
    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(
            name=expectation_suite_name + f"({dataset_type}, {data_mode})",
        ),
    )

    return df, context, data_source, data_asset, batch_definition, suite


def add_column_not_null_expectations(
    suite: ExpectationSuite,
    columns: list[str],
) -> None:
    """Adds ExpectColumnValuesToNotBeNull expectations to the suite.

    This function iterates through a list of specified columns and adds a
    non-null expectation for each one to the provided Great Expectations
    (GX) Expectation Suite. This ensures that no missing values are allowed
    in the designated columns during data validation.

    Args:
        suite (ExpectationSuite): The Great Expectations suite to which
                                  the expectations will be added.
        columns (list[str]): A list of column names that must not contain
                             null (missing) values.

    Returns:
        None
    """
    for column in columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column,
            ),
        )


def add_column_range_expectations(
    suite: ExpectationSuite,
    column_ranges: dict[str, tuple[float, float]],
) -> None:
    """Adds ExpectColumnValuesToBeBetween expectations to the suite.

    This function iterates through a dictionary of columns and their
    respective minimum and maximum allowed values. For each entry, it
    adds a Great Expectations (GX) range expectation to the provided
    suite, ensuring that all values within the column fall
    between the specified bounds.

    Args:
        suite (ExpectationSuite): The Great Expectations suite to which
                                  the expectations will be added.
        column_ranges (dict[str, tuple[float, float]]):
            A dictionary where keys are the column names and values are
            tuples defining the minimum and maximum allowed values, respectively.

    Returns:
        None
    """
    for column, (min_val, max_val) in column_ranges.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=min_val,
                max_value=max_val,
            ),
        )


def add_column_existence_expectations(
    suite: ExpectationSuite,
    column_indexes: dict[str, int],
) -> None:
    """Adds ExpectColumnToExist expectations to the suite.

    This function iterates through a dictionary specifying column names and
    their expected ordinal index (position). For each entry, it adds a Great
    Expectations (GX) expectation to the provided suite, ensuring that the
    column not only exists but is also found at the specific position within
    the DataFrame.

    Args:
        suite (ExpectationSuite): The Great Expectations suite to which
                                  the expectations will be added.
        column_indexes (dict[str, int]):
            A dictionary where keys are the column names and values are
            their positions in the DataFrame.

    Returns:
        None: The function modifies the provided ExpectationSuite object
            in place and returns nothing.
    """
    for column, index in column_indexes.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(
                column=column,
                column_index=index,
            ),
        )


def add_column_type_expectations(
    suite: ExpectationSuite,
    column_types: dict[str, str],
) -> None:
    """Adds ExpectColumnValuesToBeOfType expectations to the suite.

    This function iterates through a dictionary specifying column names and
    their expected data types. For each entry, it adds a Great Expectations
    (GX) type expectation to the provided suite ensuring that all non-missing
    values within the column conform to the defined data type.

    Args:
        suite (ExpectationSuite): The Great Expectations suite to which
                                  the expectations will be added.
        column_types (dict[str, str]):
            A dictionary where keys are the column names and values are
            the expected column data type.

    Returns:
        None
    """
    for column, expected_type in column_types.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=column,
                type_=expected_type,
            ),
        )


def add_column_count_expectation(
    suite: ExpectationSuite,
    column_count: int,
) -> None:
    """Adds ExpectTableColumnCountToEqual expectation to the suite.

    This function adds a Great Expectations (GX) expectation that verifies
    the total number of columns in the DataFrame matches the length of the
    provided list of expected columns. This ensures the structural integrity
    of the dataset header.

    Args:
        suite (ExpectationSuite): The Great Expectations suite to which
                                  the expectation will be added.
        column_count (int): The expected column count.

    Returns:
        None
    """
    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToEqual(value=column_count),
    )


def add_column_order_expectation(
    suite: ExpectationSuite,
    columns: list[str],
) -> None:
    """Adds ExpectTableColumnsToMatchOrderedList expectation to the suite.

    This function adds a Great Expectations (GX) expectation that strictly
    verifies if the list of columns in the DataFrame matches exactly
    the provided list of expected columns, including their precise order
    and count.

    Args:
        suite (ExpectationSuite): The Great Expectations suite to which
                                  the expectation will be added.
        columns (list[str]): The ordered list of column names
                             that the DataFrame must contain.

    Returns:
        None
    """
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=columns,
        ),
    )


def add_row_count_expectation(
    suite: ExpectationSuite,
    row_count: int,
) -> None:
    """Adds ExpectTableRowCountToEqual expectation to the suite.

    This function adds a Great Expectations (GX) expectation that strictly
    verifies if the total number of rows in the DataFrame matches exactly
    the specified expected count.

    Args:
        suite (ExpectationSuite): The Great Expectations suite to which
                                  the expectation will be added.
        row_count (int): The expected total number of rows that the
                         DataFrame must contain.

    Returns:
        None
    """
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToEqual(
            value=row_count,
        ),
    )


def run_dataset_testing(
    context: EphemeralDataContext,
    batch_definition: BatchDefinition,
    suite: ExpectationSuite,
    df: pd.DataFrame,
    validation_definition_name: str = DATA_TESTING_VALIDATION_DEFINITION_NAME,
    checkpoint_name: str = DATA_TESTING_CHECKPOINT_NAME,
) -> None:
    """Executes the data validation process using Great Expectations.

    This function orchestrates the final steps of data quality testing:
    defining the Validation Definition, setting up a Checkpoint, and
    running the validation process against the loaded DataFrame. It prints
    a summary of the results and asserts that all expectations in the
    suite were successfully met.

    Args:
        context (EphemeralDataContext): The active Great Expectations Data Context.
        batch_definition (BatchDefinition): The definition specifying the data
                                            to validate.
        suite (ExpectationSuite): The suite containing all defined data quality
                                  expectations.
        df (pd.DataFrame): The DataFrame containing the data to be validated.
        validation_definition_name (str): Name to assign to the Validation Definition.
        checkpoint_name (str): Name to assign to the Checkpoint.

    Returns:
        None
    """
    # Define validation definition
    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name=validation_definition_name,
            data=batch_definition,
            suite=suite,
        ),
    )

    # Define a checkpoint
    checkpoint = context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name=checkpoint_name,
            validation_definitions=[validation_definition],
        ),
    )

    # Run checkpoint over dataframe, show the results,
    # and assert everything went well
    checkpoint_result = checkpoint.run(
        batch_parameters={
            DATA_TESTING_CHECKPOINT_BATCH_PARAMETERS_DATAFRAME_KEY_NAME: df,
        },
    )
    print(checkpoint_result.describe())
    assert checkpoint_result.success is True
