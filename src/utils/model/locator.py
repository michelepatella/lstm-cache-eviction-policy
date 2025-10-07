from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    TRAINED_MODEL_DYNAMIC_DIRECTORY,
    TRAINED_MODEL_DYNAMIC_FILE_NAME,
    TRAINED_MODEL_STATIC_DIRECTORY,
    TRAINED_MODEL_STATIC_FILE_NAME,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def get_model_path(data_distribution_mode: str) -> str:
    """
    Retrieve the model path.

    This function retrieves the model path
    dynamically, based on data distribution mode.

    Parameters:
        data_distribution_mode (str): Data distribution mode selected.

    Returns:
        str: Model path.
    """
    debug(
        f"Data distribution mode to determine "
        f"model path: {data_distribution_mode}"
    )

    # Define model path according to
    # data distribution mode
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        model_path = (
            TRAINED_MODEL_STATIC_DIRECTORY / TRAINED_MODEL_STATIC_FILE_NAME
        )
    else:
        model_path = (
            TRAINED_MODEL_DYNAMIC_DIRECTORY / TRAINED_MODEL_DYNAMIC_FILE_NAME
        )

    info(f"Model path retrieved: {model_path}")

    return model_path
