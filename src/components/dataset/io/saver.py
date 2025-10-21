import pandas as pd

from components.const import DATASET_INDEX_DISABLED
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def save_dataset(
    df: pd.DataFrame, path: str, index: bool = DATASET_INDEX_DISABLED
) -> None:
    """
    Save Pandas dataframe.

    This function saves Pandas dataframe
    to CSV dataset at specified path.

    Args:
        df (pd.DataFrame): Pandas dataframe to be saved.
        path (str): Path to save dataset to.
        index (bool): If True, save dataset index. Otherwise, don't.

    Returns:
        None

    Raises:
        RuntimeError: If saving the dataset fails:
            * File cannot be written due to I/O error (OSError).
    """
    try:
        debug(f"Path to save dataset to: {path}")

        # Convert Pandas dataframe
        # to CSV file, and save it to
        # given path
        df.to_csv(path, index=index)

        info(f"Dataset saved at: {path}")
    except OSError as e:
        msg = "Failed to save dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
