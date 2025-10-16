import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset.

    This function loads an existing dataset
    from a provided path and returns it as a
    pandas DataFrame.

    Args:
        path (str): Path to load dataset from.

    Returns:
        pd.DataFrame: Dataset loaded.

    Raises:
        RuntimeError: If an error occurs while loading the dataset, e.g.:
            * Generic I/O error.
            * The dataset file is empty.
            * An error occurred while parsing the dataset file.
    """
    debug(f"Path to load dataset from: {path}")

    try:
        # Load dataset from
        # retrieved path
        df = pd.read_csv(path)

        debug(f"Shape of dataset loaded: {df.shape}")

        info(f"Dataset loaded from: {path}")

        return df
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as e:
        msg = "Failed to load dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
