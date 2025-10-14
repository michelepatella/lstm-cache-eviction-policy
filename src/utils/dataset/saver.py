import pandas as pd

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """
    Save Pandas dataframe.

    This function saves Pandas dataframe
    to CSV dataset at specified path.

    Args:
        df (pd.DataFrame): Pandas dataframe to be saved.
        path (str): Path to save dataset to.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while saving the dataset e.g.:
            * If a generic operating system error occurs.
    """
    debug(f"Path to save dataset to: {path}")

    try:
        # Convert Pandas dataframe
        # to CSV file, and save it to
        # given path
        df.to_csv(path, index=False)
    except OSError as e:
        msg = "Failed to save dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Dataset saved at {path}")
