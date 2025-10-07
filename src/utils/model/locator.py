from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    TRAINED_MODEL_DYNAMIC_DIRECTORY,
    TRAINED_MODEL_DYNAMIC_FILE_NAME,
    TRAINED_MODEL_STATIC_DIRECTORY,
    TRAINED_MODEL_STATIC_FILE_NAME,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def get_model_abs_path(data_distribution_mode: str) -> str:
    """
    Retrieve the model absolute path.

    This function retrieves the model absolute path
    dynamically, based on data distribution mode.

    Parameters:
        data_distribution_mode (str): Data distribution mode selected.

    Returns:
        str: Model absolute path.
    """
    debug(
        f"Data distribution mode to determine "
        f"model absolute path: {data_distribution_mode}"
    )

    # Define model path according to
    # data distribution mode
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        model_abs_path = (
            TRAINED_MODEL_STATIC_DIRECTORY / TRAINED_MODEL_STATIC_FILE_NAME
        )
    else:
        model_abs_path = (
            TRAINED_MODEL_DYNAMIC_DIRECTORY / TRAINED_MODEL_DYNAMIC_FILE_NAME
        )

    info(f"Model absolute path retrieved: {model_abs_path}")

    return model_abs_path
