import pandas as pd

from pipeline.const import (
    SIN_TIME_COLUMN_NAME,
    COS_TIME_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
    REQUEST_COLUMN_NAME,
)
from utils.dataset.columns.setter import set_dataset_column
from utils.dataset.columns.dropper import drop_dataset_column
from utils.dataset.columns.reorderer import reorder_dataset_columns
from utils.time.encoding_decoding.trig_encoder import encode_time_trigonometrically
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
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
        RuntimeError: If an error occurs while building features, e.g.:
            * If the time or target column does not exist in the dataset.
            * If the dataframe or columns are not of the expected type.
    """
    try:
        # Retrieve time column and convert it to
        # numpy array
        time_column_array = df[TIMESTAMP_COLUMN_NAME].to_numpy()

        # Encode time trigonometrically
        sin_time, cos_time = encode_time_trigonometrically(time_column_array)

        # Add new columns
        df = set_dataset_column(df, SIN_TIME_COLUMN_NAME, sin_time)
        df = set_dataset_column(df, COS_TIME_COLUMN_NAME, cos_time)

        # Drop the original time column
        df = drop_dataset_column(df, TIMESTAMP_COLUMN_NAME)

        # Reorder columns so that target is last
        df = reorder_dataset_columns(df, REQUEST_COLUMN_NAME)

        info("Dataset features built")

        return df
    except (KeyError, TypeError) as e:
        msg = "Failed to build dataset features"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
