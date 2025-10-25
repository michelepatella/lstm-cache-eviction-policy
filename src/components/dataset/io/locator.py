from components.const import (
    DATASET_DYNAMIC_PROCESSED_FILE_PATH,
    DATASET_DYNAMIC_RAW_FILE_PATH,
    DATASET_STATIC_PROCESSED_FILE_PATH,
    DATASET_STATIC_RAW_FILE_PATH,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from src.const import DATA_DISTRIBUTION_STATIC_MODE, DATASET_RAW_TYPE


def get_dataset_abs_path(
    dataset_type: str, data_distribution_mode: str
) -> str:
    """
    Retrieve the dataset absolute path.

    This function retrieves the dataset absolute path dynamically,
    based on data distribution mode.

    Args:
        dataset_type (str): Type of dataset requested (raw or preprocessed).
        data_distribution_mode (str): Data distribution mode selected.

    Returns:
        str: Dataset absolute path.

    Raises:
        RuntimeError: If configuration object has
                      unexpected structure.

    Raises:
        RuntimeError: If retrieving the dataset path fails:
            * Misconfigured directory or wrong filename constants (AttributeError).
    """
    try:
        debug(
            "Dataset absolute path retrieval started",
            extra={
                "dataset_type": dataset_type,
                "data_distribution_mode": data_distribution_mode,
                "context": "Dataset absolute path retrieval",
            },
        )

        # Determine dataset path based on
        # data distribution mode
        if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
            # For static data distribution mode
            if dataset_type == DATASET_RAW_TYPE:
                # For raw dataset
                dataset_abs_path = DATASET_STATIC_RAW_FILE_PATH
            else:
                # For preprocessed dataset
                dataset_abs_path = DATASET_STATIC_PROCESSED_FILE_PATH
        else:
            # For dynamic data distribution mode
            if dataset_type == DATASET_RAW_TYPE:
                # For raw dataset
                dataset_abs_path = DATASET_DYNAMIC_RAW_FILE_PATH
            else:
                # For preprocessed dataset
                dataset_abs_path = DATASET_DYNAMIC_PROCESSED_FILE_PATH

        debug(
            "Dataset absolute path retrieval completed",
            extra={
                "dataset_type": dataset_type,
                "data_distribution_mode": data_distribution_mode,
                "dataset_abs_path": dataset_abs_path,
                "context": "Dataset absolute path retrieval",
            },
        )

        return dataset_abs_path
    except AttributeError as e:
        msg = "Dataset absolute path retrieval failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "dataset_type": dataset_type,
                "data_distribution_mode": data_distribution_mode,
                "context": "Dataset absolute path retrieval",
            },
        )
        raise RuntimeError(msg) from e
