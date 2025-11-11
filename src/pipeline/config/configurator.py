"""configurator.py

Utility module for loading, validating, and preparing the pipeline's
configuration settings from a YAML file.

This module provides the `prepare_config` function, which serves as the
central point for accessing validated configuration data, ensuring all
pipeline steps operate with consistent and correctly typed parameters
defined by the Pydantic schema.

Functions:
    prepare_pipeline_config() -> PipelineConfig
        Loads the YAML configuration file, validates it against the Config
        Pydantic model, and returns the configuration object.
"""

from components.yaml.io.loader import load_yaml
from pipeline.config.pydantic.pipeline_config import PipelineConfig
from pipeline.const import CONFIG_FILE_PATH


def prepare_pipeline_config() -> PipelineConfig:
    """Prepare configuration required to run the entire pipeline.

    This function prepares the pipeline's configuration by orchestrating
    loading and validation of YAML configuration file. If the whole YAML
    configuration file is valid, it builds and returns the class representing
    the configuration settings.

    Returns:
        PipelineConfig: Class representing the configuration settings of
                        the entire pipeline.
    """
    # Load the YAML configuration file
    pipeline_config_file = load_yaml(CONFIG_FILE_PATH)

    # Validate and parse YAML configuration file
    pipeline_config = PipelineConfig(**pipeline_config_file)

    return pipeline_config
