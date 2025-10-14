import numpy as np
import pandas as pd

from utils.logs.levels.info_logger import info


def set_dataset_column(
    df: pd.DataFrame, column_name: str, column_values: np.ndarray
) -> pd.DataFrame:
    """
    Set a column to the dataset.

    This function, given a dataframe and a column
    name along with its values, set it to the
    provided dataset.

    Args:
        df (pd.DataFrame): Dataset to update.
        column_name (str): Name of the column to set.
        column_values (np.ndarray): Values to insert into the column.

    Returns:
        pd.DataFrame: DataFrame with the column set.
    """
    # Set column to dataset
    df[column_name] = column_values

    info(f"Column set to dataset: '{column_name}'")

    return df
