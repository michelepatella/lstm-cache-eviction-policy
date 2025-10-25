from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info
from components.yaml.io.loader import load_yaml
from pipeline.config.pydantic.config import Config
from pipeline.const import CONFIG_FILE_PATH


def prepare_config() -> Config:
    """
    Prepare configuration required to run the entire pipeline.

    This function prepares the pipeline's configuration by orchestrating
    loading and validation of YAML configuration file. If the whole YAML
    configuration file is valid, it builds and returns the class representing
    the configuration settings.

    Returns:
        Config: Class representing the configuration settings of
                the entire pipeline.
    """
    # Load the YAML configuration file
    config_file = load_yaml(CONFIG_FILE_PATH)

    # Validate and parse YAML configuration file
    config = Config(**config_file)

    return config
