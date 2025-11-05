import great_expectations as gx

from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from const import (
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_COLUMN_TIMESTAMP_NAME,
    TIME_END_HOUR,
    TIME_START_HOUR, DATASET_RAW_COLUMNS, DATASET_COLUMN_TIMESTAMP_TYPE, DATASET_COLUMN_REQUEST_TYPE, DATASET_RAW_TYPE,
)
from pipeline.config.configurator import prepare_config


def test_raw_dataset() -> None:
    # ----------------------------
    # Setup
    # ----------------------------
    config = prepare_config()
    df = load_dataset(get_dataset_abs_path(DATASET_RAW_TYPE, config.data.general.mode))

    # 1. Create a data context
    context = gx.get_context()

    # 2. Define a datasource
    data_source = context.data_sources.add_pandas(name="datasource")

    # 3. Create a data asset for the dataframe
    data_asset = data_source.add_dataframe_asset(name="dataframe_asset")

    # 4. Create a batch definition and a batch to validate
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")

    # 5. Create a suite of expectations
    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(
            name="Raw dataset expectations",
        ),
    )

    # ----------------------------
    # Completeness checks
    # ----------------------------

    # Ensure no columns have NaN as value
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column=DATASET_COLUMN_TIMESTAMP_NAME,
        ),
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column=DATASET_COLUMN_REQUEST_NAME,
        ),
    )

    # ----------------------------
    # Numeric checks
    # ----------------------------

    # Ensure timestamps (in hours) are
    # within the valid range, as well as requests
    # are between min and max keys
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column=DATASET_COLUMN_TIMESTAMP_NAME,
            min_value=TIME_START_HOUR,
            max_value=TIME_END_HOUR + 1,
        ),
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column=DATASET_COLUMN_REQUEST_NAME,
            min_value=config.data.general.keys.min,
            max_value=config.data.general.keys.max,
        ),
    )

    # ----------------------------
    # Schema checks
    # ----------------------------

    # Ensure columns existence
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(
            column=DATASET_COLUMN_TIMESTAMP_NAME,
            column_index=DATASET_RAW_COLUMNS.index(DATASET_COLUMN_TIMESTAMP_NAME),
        ),
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(
            column=DATASET_COLUMN_REQUEST_NAME,
            column_index=DATASET_RAW_COLUMNS.index(DATASET_COLUMN_REQUEST_NAME),
        ),
    )

    # Ensure columns are of correct type
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column=DATASET_COLUMN_TIMESTAMP_NAME,
            type_=DATASET_COLUMN_TIMESTAMP_TYPE,
        ),
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column=DATASET_COLUMN_REQUEST_NAME,
            type_=DATASET_COLUMN_REQUEST_TYPE,
        ),
    )

    # Ensure the number of columns is correct
    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToEqual(
            value=len(DATASET_RAW_COLUMNS)
        ),
    )

    # Ensure columns are sorted as expected
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=DATASET_RAW_COLUMNS,
        ),
    )

    # ----------------------------
    # Uniqueness checks
    # ----------------------------

    # Ensure dataset has no duplicates
    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=[DATASET_COLUMN_TIMESTAMP_NAME, DATASET_COLUMN_REQUEST_NAME],
        ),
    )

    # ----------------------------
    # Volume checks
    # ----------------------------

    # Ensure dataset has expected volume
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToEqual(
            value=config.data.general.requests,
        ),
    )

    # ----------------------------
    # Suite validation
    # ----------------------------
    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="Validation definition",
            data=batch_definition,
            suite=suite
        )
    )

    checkpoint = context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name="Checkpoint",
            validation_definitions=[validation_definition],
        )
    )
    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})
    print(checkpoint_result.describe())
    assert checkpoint_result.success is True


if __name__ == "__main__":
    test_raw_dataset()