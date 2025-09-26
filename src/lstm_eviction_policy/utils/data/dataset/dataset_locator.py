from const import DATA_DISTRIBUTION_STATIC_MODE
from lstm_eviction_policy.config.classes.Config import (
    Config,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    error,
    info,
)


def get_dataset_path(config: Config) -> str:
    """
    Retrieve the dataset path.

    This function retrieves the dataset path
    dynamically, based on data distribution mode.

    Parameters:
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
            dataset_path = config.data.dataset.paths.static
        else:
            # For dynamic data distribution mode
            dataset_path = config.data.dataset.paths.dynamic

        info(f"Dataset path retrieved: {dataset_path}")

        return dataset_path
    except AttributeError as e:
        msg = "Failed to retrieve dataset path"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
