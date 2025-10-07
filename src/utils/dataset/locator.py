from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    DATASET_RAW_TYPE,
    DYNAMIC_PROCESSED_DATASET_DIRECTORY,
    DYNAMIC_PROCESSED_DATASET_FILE_NAME,
    DYNAMIC_RAW_DATASET_DIRECTORY,
    DYNAMIC_RAW_DATASET_FILE_NAME,
    STATIC_PROCESSED_DATASET_DIRECTORY,
    STATIC_PROCESSED_DATASET_FILE_NAME,
    STATIC_RAW_DATASET_DIRECTORY,
    STATIC_RAW_DATASET_FILE_NAME,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def get_dataset_path(dataset_type: str, data_distribution_mode: str) -> str:
    """
    Retrieve the dataset path.

    This function retrieves the dataset path
    dynamically, based on data distribution mode.

    Parameters:
        dataset_type (str): Type of dataset requested (raw or preprocessed).
        data_distribution_mode (str): Data distribution mode selected.

    Returns:
        str: Dataset path.

    Raises:
        RuntimeError: If configuration object has
                      unexpected structure.
    """
    try:
        debug(
            f"Data distribution mode for "
            f"loading dataset: {data_distribution_mode}"
        )

        # Determine dataset path based on
        # data distribution mode
        if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
            # For static data distribution mode
            if dataset_type == DATASET_RAW_TYPE:
                # For raw dataset
                dataset_path = (
                    STATIC_RAW_DATASET_DIRECTORY / STATIC_RAW_DATASET_FILE_NAME
                )
            else:
                # For preprocessed dataset
                dataset_path = (
                    STATIC_PROCESSED_DATASET_DIRECTORY
                    / STATIC_PROCESSED_DATASET_FILE_NAME
                )
        else:
            # For dynamic data distribution mode
            if dataset_type == DATASET_RAW_TYPE:
                # For raw dataset
                dataset_path = (
                    DYNAMIC_RAW_DATASET_DIRECTORY
                    / DYNAMIC_RAW_DATASET_FILE_NAME
                )
            else:
                # For preprocessed dataset
                dataset_path = (
                    DYNAMIC_PROCESSED_DATASET_DIRECTORY
                    / DYNAMIC_PROCESSED_DATASET_FILE_NAME
                )

        info(f"Dataset path retrieved: {dataset_path}")

        return dataset_path
    except AttributeError as e:
        msg = "Failed to retrieve dataset path"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
