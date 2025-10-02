from const import DATA_DISTRIBUTION_STATIC_MODE, DATASET_RAW_TYPE
from pipeline.config.classes.Config import (
    Config,
)
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def get_dataset_path(dataset_type: str, config: Config) -> str:
    """
    Retrieve the dataset path.

    This function retrieves the dataset path
    dynamically, based on data distribution mode.

    Parameters:
        dataset_type (str): Type of dataset requested (raw or preprocessed).
        config (Config): Configuration object.

    Returns:
        str: Dataset path.

    Raises:
        RuntimeError: If configuration object has
                      unexpected structure.
    """
    try:
        # Retrieve data distribution mode
        data_distribution_mode = config.data.general.mode

        debug(
            f"Data distribution mode for loading dataset: {data_distribution_mode}"
        )

        # Determine dataset path based on
        # data distribution mode
        if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
            # For static data distribution mode
            if dataset_type == DATASET_RAW_TYPE:
                # For raw dataset
                dataset_path = config.data.dataset.paths.raw.static
            else:
                # For preprocessed dataset
                dataset_path = config.data.dataset.paths.preprocessed.static
        else:
            # For dynamic data distribution mode
            # For static data distribution mode
            if dataset_type == DATASET_RAW_TYPE:
                # For raw dataset
                dataset_path = config.data.dataset.paths.raw.dynamic
            else:
                # For preprocessed dataset
                dataset_path = config.data.dataset.paths.preprocessed.dynamic

        info(f"Dataset path retrieved: {dataset_path}")

        return dataset_path
    except AttributeError as e:
        msg = "Failed to retrieve dataset path"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
