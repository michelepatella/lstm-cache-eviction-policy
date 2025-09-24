import pandas as pd

from lstm_eviction_policy.config.classes.Config import (
    Config,
)
from lstm_eviction_policy.utils.data.dataset.dataset_locator import (
    get_dataset_path,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    error,
    info,
)


def load_dataset(config: Config) -> pd.DataFrame:
    """
    Load existing dataset.

    This function loads an existing dataset
    from a specific path — which depends on the
    data distribution mode — and returns it as a
    pandas DataFrame.

    Parameters:
        config (Config): Configuration object.

    Returns:
        pd.DataFrame: Dataset loaded.

    Raises:
        OSError: If a generic I/O error occurred.
        pd.errors.EmptyDataError: If dataset file is empty.
        pd.errors.ParserError: If an error parsing dataset file
                               occurred.
    """
    # Retrieve path to load dataset from
    dataset_path = get_dataset_path(config)

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
