import pandas as pd

from components.logs.levels.error_logger import error


def shift_dataset_column(
    df: pd.DataFrame,
    column_name: str,
    shift: float,
) -> None:
    """Shift the values of a dataset column.

    This function applies the given shift and updates the
    dataset in-place.

    Args:
        df (pd.DataFrame): Dataset containing the column to shift.
        column_name (str): Name of the column to shift.
        shift (float): Value to add/subtract from column values.

    Returns:
        None

    Raises:
        RuntimeError: If shifting the column fails:
            * Column not found in dataset (KeyError).
            * Dataset or column values not suitable for arithmetic
              (TypeError, AttributeError).
    """
    try:
        # Cast to int and apply shift to column
        df[column_name] = df[column_name] + shift
    except (TypeError, AttributeError, KeyError) as e:
        msg = "Dataset column shifting failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "column_name": column_name,
                "shift_value": shift,
                "existing_columns": (
                    df.columns.tolist() if hasattr(df, "columns") else None
                ),
                "num_rows": len(df) if hasattr(df, "__len__") else None,
                "context": "Dataset column shifting",
            },
        )
        raise RuntimeError(msg) from e
