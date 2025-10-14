import pandas as pd

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def reorder_dataset_columns(
    df: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """
    Reorder the dataset columns.

    This function reorders the DataFrame columns so
    that the provided target column is the last one.

    Args:
        df (pd.DataFrame): Dataset to update.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: DataFrame with target column as last column.

    Raises:
        RuntimeError: If an error occurs while reordering the dataset
                      e.g.:
                        * The target column does not exist.
    """
    try:
        # Consider all the columns different
        # from the target one as features
        features = [col for col in df.columns if col != target_column]

        debug(f"Dataset feature column(s): {features}")
        debug(f"Dataset target column: {target_column}")

        # Insert the feature columns before
        # the target one
        new_df = df[features + [target_column]]

        debug(f"Reordered dataset columns: {new_df.columns}")

        info("Dataset columns reordered")

        return new_df
    except KeyError as e:
        msg = "Failed to reorder dataset columns"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
