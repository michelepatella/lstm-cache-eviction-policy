"""configurator.py

Module dedicated to loading, validating, and preparing the testing configuration.

This module ensures that the YAML configuration file for the test suite is correctly
loaded from the file system, validated against the Pydantic data model, and returned
as a structured object for use across the tests.

Functions:
    prepare_tests_config() -> TestsConfig
        Loads the test configuration from a YAML file, validates it, and returns
        the parsed Pydantic object.
"""

from components.yaml.io.loader import load_yaml
from tests.config.pydantic.tests_config import TestsConfig
from tests.const import TESTS_CONFIG_FILE_PATH


def prepare_tests_config() -> TestsConfig:
    """Loads and validates the testing configuration from the YAML file.

    This function reads the configuration file, parses its content, and uses
    Pydantic to perform schema validation, ensuring all required fields are
    present and correctly typed.

    Returns:
        TestsConfig: The validated and parsed configuration object.
    """
    tests_config_file = load_yaml(TESTS_CONFIG_FILE_PATH)

    # Validate and parse YAML tests config file
    tests_config = TestsConfig(**tests_config_file)

    return tests_config
