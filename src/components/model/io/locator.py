from components.const import (
    MODEL_TRAINED_DYNAMIC_FILE_PATH,
    MODEL_TRAINED_STATIC_FILE_PATH,
)
from components.logs.levels.debug_logger import debug
from const import DATA_DISTRIBUTION_STATIC_MODE


def get_model_abs_path(data_distribution_mode: str) -> str:
    """
    Retrieve the model absolute path.

    This function retrieves the model absolute path
    dynamically, based on data distribution mode.

    Args:
        data_distribution_mode (str): Data distribution mode set.

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
        model_abs_path = MODEL_TRAINED_STATIC_FILE_PATH
    else:
        model_abs_path = MODEL_TRAINED_DYNAMIC_FILE_PATH

    debug(f"Model absolute path: {model_abs_path}")

    return model_abs_path
