import numpy as np
import pandas as pd

from components.logs.levels.error_logger import error


def set_dataset_column(
    df: pd.DataFrame,
    column_name: str,
    column_values: np.ndarray,
) -> pd.DataFrame:
    """Set a column to the dataset.

    This function, given a dataframe and a column name along with
    its values, set it to the provided dataset.

    Args:
        df (pd.DataFrame): Dataset to update.
        column_name (str): Name of the column to set.
        column_values (np.ndarray): Values to insert into the column.

    Returns:
        pd.DataFrame: DataFrame with the column set.

    Raises:
        RuntimeError: If setting the column fails:
            * Column values length mismatch with dataset (ValueError).
            * Invalid dataframe or column values type
              (TypeError, AttributeError).
    """
    try:
        # Set column to dataset
        df[column_name] = column_values

        return df
    except (ValueError, TypeError, AttributeError) as e:
        msg = "Dataset column setting failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "column_name": column_name,
                "column_values_length": (
                    len(column_values)
                    if hasattr(column_values, "__len__")
                    else None
                ),
                "columns_existing": (
                    df.columns.tolist() if hasattr(df, "columns") else None
                ),
                "context": "Dataset column setting",
            },
        )
        raise RuntimeError(msg) from e
