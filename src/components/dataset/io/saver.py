import pandas as pd

from components.const import DATASET_INDEX_DISABLED
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def save_dataset(
    df: pd.DataFrame,
    path: str,
    index: bool = DATASET_INDEX_DISABLED,
) -> None:
    """Save Pandas dataframe.

    This function saves Pandas dataframe
    to CSV dataset at specified path.

    Args:
        df (pd.DataFrame): Pandas dataframe to be saved.
        path (str): Path to save dataset to.
        index (bool): If True, save dataset index. Otherwise, don't.

    Returns:
        None

    Raises:
        RuntimeError: If saving the dataset fails:
            * File cannot be written due to I/O error (OSError).
    """
    try:
        debug(
            "Dataset saving started",
            extra={
                "path": path,
                "save_index": index,
                "rows_num": len(df) if hasattr(df, "__len__") else None,
                "column_num": (
                    len(df.columns) if hasattr(df, "columns") else None
                ),
                "context": "Dataset saving",
            },
        )

        # Convert Pandas dataframe
        # to CSV file, and save it to
        # given path
        df.to_csv(path, index=index)

        debug(
            "Dataset saving completed",
            extra={
                "path": path,
                "save_index": index,
                "rows_num": len(df) if hasattr(df, "__len__") else None,
                "columns_num": (
                    len(df.columns) if hasattr(df, "columns") else None
                ),
                "context": "Dataset saving",
            },
        )
    except OSError as e:
        msg = "Dataset saving failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": path,
                "save_index": index,
                "rows_num": len(df) if hasattr(df, "__len__") else None,
                "columns_num": (
                    len(df.columns) if hasattr(df, "columns") else None
                ),
                "context": "Dataset saving",
            },
        )
        raise RuntimeError(msg) from e
