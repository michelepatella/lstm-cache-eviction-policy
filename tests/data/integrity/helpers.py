"""data_integrity_helpers.py

Module dedicated to preparing the necessary data and configuration structure
for executing data integrity tests.

This module provides utility function for initializing the testing environment
(loading data into Deepchecks Dataset and parsing configurations).

Functions:
    initialize_data_integrity_tests(
        dataset_type: str,
        data_mode: str,
    ) -> tuple[Dataset, TestsConfig]
        Loads the specified dataset and the tests configuration, returning both
        as Deepchecks Dataset and Pydantic TestsConfig objects.
"""

from deepchecks import Dataset

from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from tests.config.configurator import prepare_tests_config
from tests.config.pydantic.tests_config import TestsConfig
from tests.helpers.dc_helpers import create_dc_dataset


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
    # a Deepchecks Dataset object
    df = load_dataset(get_dataset_abs_path(dataset_type, data_mode))
    dc_dataset = create_dc_dataset(df)

    # Prepare the tests configuration
    tests_config = prepare_tests_config()

    return dc_dataset, tests_config
