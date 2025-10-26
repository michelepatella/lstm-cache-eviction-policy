import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def remove_dataset_duplicates(
    df: pd.DataFrame,
    subset: list[str],
) -> pd.DataFrame:
    """Remove duplicates from dataset.

    This function removes rows with duplicated values from a given
    subset of dataset column(s), returning a new clean dataset.

    Args:
        df (pd.DataFrame): Dataset to remove duplicates from.
        subset (list[str]): Column(s) of dataset to remove duplicates from.

    Returns:
        pd.DataFrame: Dataset without duplicated values.

    Raises:
        RuntimeError: If removing duplicates fails:
            * Dataset object is not a valid DataFrame (AttributeError).
            * Specified subset columns are invalid (KeyError, TypeError).
    """
    try:
        debug(
            "Dataset duplicates removal started",
            extra={
                "rows_before_num": (
                    len(df) if isinstance(df, pd.DataFrame) else None
                ),
                "subset_columns": subset,
                "context": "Dataset duplicates removal",
            },
        )

        # Remove duplicates from dataset
        new_df = df.drop_duplicates(subset=subset)

        debug(
            "Dataset duplicates removal completed",
            extra={
                "rows_after_num": len(new_df),
                "duplicates_removed_num": len(df) - len(new_df),
                "subset_columns": subset,
                "context": "Dataset duplicates removal",
            },
        )

        return new_df
    except (AttributeError, KeyError, TypeError) as e:
        msg = "Dataset duplicates removal failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "subset_columns": subset,
                "rows_num": len(df) if isinstance(df, pd.DataFrame) else None,
                "context": "Dataset duplicates removal",
            },
        )
        raise RuntimeError(msg) from e
