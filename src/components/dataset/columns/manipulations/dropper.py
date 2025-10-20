import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def drop_dataset_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Drop a column from the dataset.

    This function, given a DataFrame and a column name, removes the
    column from the dataset.

    Args:
        df (pd.DataFrame): Dataset to update.
        column_name (str): Name of the column to drop.

    Returns:
        pd.DataFrame: DataFrame with the column removed.

    Raises:
        RuntimeError: If dropping the column fails:
            * Column does not exist in the dataset (KeyError).
    """
    try:
        debug(f"Column to be dropped from dataset: '{column_name}'")

        # Drop column from dataset
        new_df = df.drop(columns=[column_name])

        debug(f"Column '{column_name}' dropped from dataset")

        return new_df
    except KeyError as e:
        msg = "Failed to remove column from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
