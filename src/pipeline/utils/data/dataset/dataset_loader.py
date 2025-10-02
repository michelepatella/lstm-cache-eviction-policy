import pandas as pd

from pipeline.config.classes.Config import (
    Config,
)
from pipeline.utils.data.dataset.dataset_locator import (
    get_dataset_path,
)
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def load_dataset(dataset_type: str, config: Config) -> pd.DataFrame:
    """
    Load existing dataset.

    This function loads an existing dataset
    from a specific path — which depends on the
    data distribution mode — and returns it as a
    pandas DataFrame.

    Parameters:
        dataset_type (str): Dataset type to be loaded.
        config (Config): Configuration object.

    Returns:
        pd.DataFrame: Dataset loaded.

    Raises:
        RuntimeError: If an error occurs while loading the dataset, e.g.:
            * Generic I/O error.
            * The dataset file is empty.
            * An error occurred while parsing the dataset file.
    """
    # Retrieve path to load dataset from
    dataset_path = get_dataset_path(dataset_type, config)

    debug(f"Path to load dataset from: {dataset_path}")

    try:
        # Load dataset from
        # retrieved path
        df = pd.read_csv(dataset_path)

        debug(f"Shape of dataset loaded: {df.shape}")

        info(f"Dataset loaded from: {dataset_path}")

        return df
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as e:
        msg = "Failed to load dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
