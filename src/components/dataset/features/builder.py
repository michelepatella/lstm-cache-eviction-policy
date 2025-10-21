import pandas as pd

from components.const import (
    DATASET_COS_TIME_COLUMN_NAME,
    DATASET_SIN_TIME_COLUMN_NAME,
)
from components.dataset.columns.manipulations.dropper import (
    drop_dataset_column,
)
from components.dataset.columns.manipulations.reorderer import (
    reorder_dataset_columns,
)
from components.dataset.columns.manipulations.setter import set_dataset_column
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.time.transforms.trig_encoder import (
    encode_time_trigonometrically,
)
from src.const import DATASET_REQUEST_COLUMN_NAME, DATASET_TIMESTAMP_COLUMN_NAME


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset features with time
    encoding and target reordering.

    This function applies trigonometric encoding to the
    time column, creating sine and cosine features to represent
    cyclical time. It then reorders the columns so that the
    target column appears last, keeping feature columns first.

    Args:
        df (pd.DataFrame): Dataset containing the time and target columns.

    Returns:
        pd.DataFrame: New dataset with trigonometric time features
                      and target column reordered as the last column.

    Raises:
        RuntimeError: If building features fails:
            * Time column not found (KeyError).
            * Dataset or column values not suitable for processing (TypeError).
    """
    try:
        # Retrieve time column and convert it to
        # numpy array
        time_column_array = df[DATASET_TIMESTAMP_COLUMN_NAME].to_numpy()

        # Encode time trigonometrically
        sin_time, cos_time = encode_time_trigonometrically(time_column_array)

        # Add new columns
        df = set_dataset_column(df, DATASET_SIN_TIME_COLUMN_NAME, sin_time)
        df = set_dataset_column(df, DATASET_COS_TIME_COLUMN_NAME, cos_time)

        # Drop the original time column
        df = drop_dataset_column(df, DATASET_TIMESTAMP_COLUMN_NAME)

        # Reorder columns so that target is last
        df = reorder_dataset_columns(df, DATASET_REQUEST_COLUMN_NAME)

        debug("Dataset features built")

        return df
    except (KeyError, TypeError) as e:
        msg = "Failed to build dataset features"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
