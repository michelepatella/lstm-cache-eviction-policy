import pandas as pd

from components.logs.levels.error_logger import error


def reorder_dataset_columns(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Reorder the dataset columns.

    This function reorders the DataFrame columns so that the
    provided target column is the last one.

    Args:
        df (pd.DataFrame): Dataset to update.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: DataFrame with target column as last column.

    Raises:
        RuntimeError: If reordering columns fails:
            * Target column does not exist in the dataset (KeyError).
    """
    try:
        # Consider all the columns different
        # from the target one as features
        features = [col for col in df.columns if col != target_column]

        # Insert the feature columns before
        # the target one
        new_df = df[features + [target_column]]

        return new_df
    except KeyError as e:
        msg = "Dataset columns reordering failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "target_column": target_column,
                "columns_available": (
                    df.columns.tolist() if hasattr(df, "columns") else None
                ),
                "context": "Dataset columns reordering",
            },
        )
        raise RuntimeError(msg) from e
