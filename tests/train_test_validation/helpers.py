"""helpers.py

Module containing helper functions for setting up the testing environment
for train-test-validation tests using Deepchecks.

This module is responsible for loading the processed data splits (training,
validation, and testing), parsing the necessary configurations, and converting
the dataframes into Deepchecks Dataset objects, including the addition of
a temporary index column required for certain tests.

Functions:
    initialize_train_test_validation_tests() -> tuple[
        Dataset,
        Dataset,
        Dataset,
        TestsConfig
    ]
        Prepares the environment by loading data splits and configurations,
        returning them as Deepchecks Datasets and Pydantic configuration object.
"""

from deepchecks.tabular import Dataset

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.training_validation_splitter import (
    split_training_validation_sets,
)
from const import (
    DATASET_TESTING_SPLIT_TYPE,
    DATASET_TRAINING_SPLIT_TYPE,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from pipeline.const import DATASET_PROCESSED_TYPE
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import DATASET_COLUMN_TEMP_INDEX_NAME
from tests.helpers import create_dc_dataset


def initialize_train_test_validation_tests() -> tuple[
    Dataset,
    Dataset,
    Dataset,
    TestsConfig,
]:
    """Loads and prepares all data splits and configurations for
    train-test-validation tests.

    This function orchestrates the setup of the train test validation
    testing environment. It begins by loading the global pipeline
    configuration alongside the specific tests' configuration. Next, it
    retrieves the processed training and testing datasets, and then splits
    the training data to create the validation subset using the configured ratio.
    All three resulting dataframes (training, validation, testing) are subsequently
    converted into Deepchecks Dataset objects. Finally, a necessary temporary index
    column is added to each Deepchecks Dataset to facilitate index-based consistency
    checks.

    Returns:
        tuple[Dataset, Dataset, Dataset, PipelineConfig, TestsConfig]:
            - dc_training_set (Dataset): Training data as a Deepchecks Dataset.
            - dc_validation_set (Dataset): Validation data as a Deepchecks Dataset.
            - dc_testing_set (Dataset): Testing data as a Deepchecks Dataset.
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

    # Convert sets to DeepChecks datasets,
    # adding a temporary index column to all
    # them for further tests
    dc_training_set = create_dc_dataset(
        training_set.dataset.data.copy().assign(
            **{
                DATASET_COLUMN_TEMP_INDEX_NAME: range(
                    len(training_set.dataset.data),
                ),
            },
        ),
        index_name=DATASET_COLUMN_TEMP_INDEX_NAME,
    )
    dc_validation_set = create_dc_dataset(
        validation_set.dataset.data.copy().assign(
            **{
                DATASET_COLUMN_TEMP_INDEX_NAME: range(
                    len(validation_set.dataset.data),
                ),
            },
        ),
        index_name=DATASET_COLUMN_TEMP_INDEX_NAME,
    )
    dc_testing_set = create_dc_dataset(
        testing_set.data.copy().assign(
            **{DATASET_COLUMN_TEMP_INDEX_NAME: range(len(testing_set.data))},
        ),
        index_name=DATASET_COLUMN_TEMP_INDEX_NAME,
    )

    return (
        dc_training_set,
        dc_validation_set,
        dc_testing_set,
        tests_config,
    )
