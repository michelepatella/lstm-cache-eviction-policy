import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset.

    This function loads an existing dataset
    from a provided path and returns it as a
    pandas DataFrame.

    Args:
        path (str): Path to load dataset from.

    Returns:
        pd.DataFrame: Dataset loaded.

    Raises:
        RuntimeError: If an error occurs while loading the dataset:
            * Generic I/O errors (OSError).
            * The dataset file is empty (pd.errors.EmptyDataError).
            * Parsing errors while reading the dataset (pd.errors.ParserError).
    """
    try:
        debug(
            "Dataset loading started",
            extra={
                "dataset_path": path,
                "context": "Dataset loading",
            },
        )

        # Load dataset from
        # retrieved path
        df = pd.read_csv(path)

        debug(
            "Dataset loading completed",
            extra={
                "dataset_path": path,
                "rows_num": len(df),
                "columns_num": len(df.columns),
                "columns": df.columns.tolist(),
                "context": "Dataset loading",
            },
        )

        return df
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as e:
        msg = "Dataset loading failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "dataset_path": path,
                "context": "Dataset loading",
            },
        )
        raise RuntimeError(msg) from e
