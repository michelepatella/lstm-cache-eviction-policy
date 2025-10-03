import pandas as pd

from pipeline.config.classes.Config import Config
from utils.dataset.locator import get_dataset_path
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def save_dataset(df: pd.DataFrame, dataset_type: str, config: Config) -> None:
    """
    Save Pandas dataframe to CSV dataset.

    This function saves Pandas dataframe
    to CSV dataset at specified path.

    Parameters:
        df (pd.DataFrame): Pandas dataframe to be saved as
                           CSV dataset.
        dataset_type (str): Dataset type to be saved.
        config (Config): Configuration object.

    Returns:
        None

    Raises:
        RuntimeError: Generic operating system error while
                      saving the dataset at specified path.
    """
    # Retrieve path where
    # to save dataset
    dataset_path = get_dataset_path(dataset_type, config)

    debug(f"Path where to save dataset: {dataset_path}")

    try:
        # Convert Pandas dataframe
        # to CSV file, and save it to
        # retrieved path
        df.to_csv(dataset_path, index=False)
    except OSError as e:
        msg = "Failed to save dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Dataset saved at {dataset_path}")
