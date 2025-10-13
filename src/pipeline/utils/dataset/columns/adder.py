import numpy as np
import pandas as pd

from pipeline.utils.logs.levels.info_logger import info


def add_dataset_column(
    df: pd.DataFrame, column_name: str, column_values: np.ndarray
) -> pd.DataFrame:
    """
    Add a new column to the dataset.

    This function, given a dataframe and a column
    name along with its values, add it to the
    provided dataset.

    Args:
        df (pd.DataFrame): Dataset to update.
        column_name (str): Name of the new column.
        column_values (np.ndarray): Values to insert into the column.

    Returns:
        pd.DataFrame: DataFrame with the new column added.
    """
    # Add column to dataset
    df[column_name] = column_values

    info(f"Column added to dataset: '{column_name}'")

    return df
