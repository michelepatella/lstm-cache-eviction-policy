import pandas as pd

from lstm_eviction_policy.data_preprocessing.features_engineering.features_encoder import (
    encode_time_trigonometrically,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    error,
    info,
)


def build_features(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Build dataset features with time
    encoding and target reordering.

    This function applies trigonometric encoding to the specified
    time column, creating sine and cosine features to represent
    cyclical time. It then reorders the columns so that the
    target column appears last, keeping feature columns first.

    Parameters:
        df (pd.DataFrame): Dataset containing the time and target columns.
        time_column (str): Name of the time column to encode trigonometrically.
        target_column (str): Name of the target column to place at the end.

    Returns:
        pd.DataFrame: New dataset with trigonometric time features
                      and target column reordered as the last column.

    Raises:
        KeyError: If the time or target column does not exist in the dataset.
        TypeError: If the dataframe or columns are not of the expected type.
    """
    try:
        # Encode time trigonometrically
        new_df = encode_time_trigonometrically(df, time_column)

        # Reorder columns so that
        # target column is the last one
        features = [col for col in new_df.columns if col != target_column]
        new_df = new_df[features + [target_column]]

        debug(f"Dataset feature columns: {features}")
        debug(f"Dataset target column: {target_column}")
        debug(f"Dataset columns after re-ordering: {new_df.columns}")

        info("Dataset features built")

        return new_df
    except (KeyError, TypeError) as e:
        msg = "Failed to build dataset features"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
