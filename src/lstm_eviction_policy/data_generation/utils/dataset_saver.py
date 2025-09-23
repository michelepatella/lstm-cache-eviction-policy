import pandas as pd

from lstm_eviction_policy.utils.logs.log_utils import debug, error, info


def save_dataset(df: pd.DataFrame, dataset_path: str) -> None:
    """
    Save Pandas dataframe to CSV dataset.

    This function saves Pandas dataframe
    to CSV dataset at specified path.

    Parameters:
        df (pd.DataFrame): Pandas dataframe to be saved as
                           CSV dataset.
        dataset_path (str): Path where to save the dataset.

    Returns:
        None

    Raises:
        OSError: Generic operating system error while
                 saving the dataset at specified path.
    """
    debug(f"Path where to save the dataset: {dataset_path}")

    try:
        # Convert Pandas dataframe
        # to CSV file, and save it to
        # retrieved path
        df.to_csv(dataset_path, index=False)
    except OSError as e:
        msg = f"Failed to save dataset at {dataset_path}"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Dataset saved at {dataset_path}")
