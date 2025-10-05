from const import LOGS_CONFIGURATION_PHASE
from config.classes.Config import Config
from config.io.loader import load_config
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

    # Load the YAML configuration file
    config_file = load_config()

    # Validate and parse YAML configuration file
    config = Config(**config_file)

    info("Configuration loading and validation completed")

    return config
