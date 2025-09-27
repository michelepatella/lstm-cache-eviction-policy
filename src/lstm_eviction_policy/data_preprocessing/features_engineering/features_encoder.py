import numpy as np
import pandas as pd

from const import (
    COS_TIME_COLUMN_NAME,
    PERIOD,
    SIN_TIME_COLUMN_NAME,
)
from lstm_eviction_policy.utils.logs.levels.debug_logger import debug
from lstm_eviction_policy.utils.logs.levels.error_logger import error
from lstm_eviction_policy.utils.logs.levels.info_logger import info


def encode_time_trigonometrically(
    df: pd.DataFrame, time_column: str
) -> pd.DataFrame:
    """
    Encode a time column trigonometrically.

    This function converts a time column into
    two new features using sine and cosine transformations,
    allowing cyclical representation of time in a dataset.
    The original time column is dropped and replaced by the
    two new columns.

    Parameters:
        df (pd.DataFrame): Dataset containing the time column to encode.
        time_column (str): Name of the time column to be transformed.

    Returns:
        pd.DataFrame: New dataset with time column replaced by
                      sine and cosine features.

    Raises:
        RuntimeError: If an error occurs while encoding time trigonometrically, e.g.:
            * If the specified time column does not exist in the dataset.
            * If the time column contains non-numeric values.
            * If the period is set to zero.
    """
    try:
        debug(f"Time column to be encoded trigonometrically: {time_column}")
        debug(
            f"(Time before normalization) Min: {df[time_column].min()}, Max: {df[time_column].max()}"
        )

        # Normalize time to [0, 2pi] so that
        # to have time in cycle
        time_in_cycle = (df[time_column] % PERIOD) / PERIOD

        debug(
            f"(Time after normalization) Min: {time_in_cycle.min()}, Max: {time_in_cycle.max()}"
        )

        # Use normalized time in cycle
        # to get angles in radians
        angles = time_in_cycle * 2 * np.pi

        debug(f"Angles (radians) min: {angles.min()}, max: {angles.max()}")

        # Create new columns — sin and cosine
        # for the angles
        df[SIN_TIME_COLUMN_NAME] = np.sin(angles)
        df[COS_TIME_COLUMN_NAME] = np.cos(angles)

        # Drop the original time column,
        # use only the new trigonometric columns
        # in the new dataset
        new_df = df.drop(columns=[time_column])

        debug(
            f"Dataset columns after trigonometric time encoding: {new_df.columns}"
        )

        info("Time encoded trigonometrically")

        return new_df
    except (
        KeyError,
        TypeError,
        ZeroDivisionError,
    ) as e:
        msg = "Failed to encode time trigonometrically"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
