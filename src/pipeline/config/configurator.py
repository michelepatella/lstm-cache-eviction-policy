from pipeline.config.classes.Config import Config
from utils.yaml.loader import load_yaml
from pipeline.const import LOGS_CONFIGURATION_PHASE, CONFIG_FILE_NAME, CONFIG_DIRECTORY_PATH
from utils.logs.initializer import logs_phase
from utils.logs.levels.info_logger import info


def prepare_config() -> Config:
    """
    Prepare configuration required to run
    the entire pipeline.

    This function prepares the pipeline's configuration
    by orchestrating loading and validation of YAML configuration file.
    If the whole YAML configuration file is valid, it builds
    and returns the class representing the configuration settings.

    Returns:
        Config: Class representing the configuration settings of
                the entire pipeline.
    """
    # Set the new state
    logs_phase.set(LOGS_CONFIGURATION_PHASE)

    # Get the absolute path of the YAML
    # configuration file
    abs_config_path = CONFIG_DIRECTORY_PATH / CONFIG_FILE_NAME

    # Load the YAML configuration file
    config_file = load_yaml(abs_config_path)

    # Validate and parse YAML configuration file
    config = Config(**config_file)

    info("Configuration loading and validation completed")

    return config
