"""builder.py

Module for constructing dataset features including trigonometric time encoding,
local frequency and recency features, and target column reordering.

This module provides the `build_features` function, which:
    - Encodes the timestamp column into sine and cosine features for cyclical
      time representation.
    - Computes local frequency and local recency for each request key within
      a rolling window of a specified sequence length.
    - Adds the computed features to the dataset.
    - Drops the original timestamp column.
    - Reorders the dataset columns so that the target column is last.

Functions:
    build_features(df: pd.DataFrame, seq_len: int) -> pd.DataFrame
        Builds dataset features including time encoding, local frequency,
        local recency, and reorders columns so that the target column is last.
"""

import pandas as pd

from components.const import (
    DATASET_COLUMN_COS_TIME_NAME,
    DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
    DATASET_COLUMN_LOCAL_RECENCY_NAME,
    DATASET_COLUMN_SIN_TIME_NAME,
)
from components.dataset.columns.dropper import (
    drop_dataset_column,
)
from components.dataset.columns.reorderer import (
    reorder_dataset_columns,
)
from components.dataset.columns.setter import set_dataset_column
from components.dataset.features.derived.local_frequencies_calculator import (
    calculate_local_frequencies,
)
from components.dataset.features.derived.local_recencies_calculator import (
    calculate_local_recencies,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.time.transforms.trig_encoder import (
    encode_time_trigonometrically,
)
from src.const import (
    DATASET_COLUMN_REQUEST_NAME,
    DATASET_COLUMN_TIMESTAMP_NAME,
)


def build_features(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """Build dataset features with time
    encoding and target reordering.

    This function applies trigonometric encoding to the
    time column, creating sine and cosine features to represent
    cyclical time. Then, it computes local frequencies and recencies
    of keys, according to the provided sequence length. It finally
    reorders the columns so that the target column appears last,
    keeping feature columns first.

    Args:
        df (pd.DataFrame): Dataset containing the time and target columns.
        seq_len (int): Length of the model sequences.

    Returns:
        pd.DataFrame: New dataset with trigonometric time features,
                      local frequencies and recencies, and target
                      column reordered as the last column.

    Raises:
        RuntimeError: If building features fails:
            * Time column not found (KeyError).
            * Dataset or column values not suitable for processing (TypeError).
    """
    try:
        debug(
            "Dataset features building started",
            extra={
                "seq_len": seq_len,
                "rows_num": len(df) if hasattr(df, "__len__") else None,
                "columns_existing": (
                    df.columns.tolist() if hasattr(df, "columns") else None
                ),
                "time_column": DATASET_COLUMN_TIMESTAMP_NAME,
                "target_column": DATASET_COLUMN_REQUEST_NAME,
                "context": "Dataset features building",
            },
        )

        # Retrieve time column and convert it to
        # numpy array
        time_column_array = df[DATASET_COLUMN_TIMESTAMP_NAME].to_numpy()

        # Encode time trigonometrically and add these
        # new columns to the dataset
        sin_time, cos_time = encode_time_trigonometrically(time_column_array)
        df = set_dataset_column(df, DATASET_COLUMN_SIN_TIME_NAME, sin_time)
        df = set_dataset_column(df, DATASET_COLUMN_COS_TIME_NAME, cos_time)

        # Drop the original time column
        df = drop_dataset_column(df, DATASET_COLUMN_TIMESTAMP_NAME)

        # Construct local frequencies for each key and
        # add this new column to the dataset
        local_frequencies = calculate_local_frequencies(
            df[DATASET_COLUMN_REQUEST_NAME],
            seq_len,
        )
        df = set_dataset_column(
            df,
            DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
            local_frequencies,
        )

        # Construct local recencies for each key
        # and add this new column to the dataset
        local_recencies = calculate_local_recencies(
            df[DATASET_COLUMN_REQUEST_NAME],
            seq_len,
        )
        df = set_dataset_column(
            df,
            DATASET_COLUMN_LOCAL_RECENCY_NAME,
            local_recencies,
        )

        # Reorder columns so that target is last
        df = reorder_dataset_columns(df, DATASET_COLUMN_REQUEST_NAME)

        debug(
            "Dataset features building completed",
            extra={
                "rows_num": len(df),
                "columns_after_processing": df.columns.tolist(),
                "context": "Dataset features building",
            },
        )

        return df
    except (KeyError, TypeError) as e:
        msg = "Dataset features building failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "columns_existing": (
                    df.columns.tolist() if hasattr(df, "columns") else None
                ),
                "time_column_expected": DATASET_COLUMN_TIMESTAMP_NAME,
                "target_column_expected": DATASET_COLUMN_REQUEST_NAME,
                "context": "Dataset features building",
            },
        )
        raise RuntimeError(msg) from e
