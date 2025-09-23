from const import PIPELINE_PHASE_CONFIGURATION
from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.config.config_io.config_loader import load_config
from lstm_eviction_policy.utils.logs.log_utils import info, phase_var


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
    # Set the new pipeline state
    phase_var.set(PIPELINE_PHASE_CONFIGURATION)

    # Load the YAML configuration file
    config_file = load_config()

    # Validate and parse YAML configuration file
    config = Config(**config_file)

    info("Configuration loaded and validated")

    return config
