import pandas as pd

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error


def report_dataset_cleaning(
    original_df: pd.DataFrame,
    new_df: pd.DataFrame,
    context: str,
) -> None:
    """
    Report dataset cleaning information.

    This function reports dataset cleaning
    intermediate results to keep track of number
    of rows removed from dataset after applying
    a specific cleaning operation.

    Parameters:
        original_df (pd.DataFrame): Original dataframe (i.e., before
                                    applying the cleaning operation).
        new_df (pd.DataFrame): New dataframe (i.e., after applying
                               the cleaning operation).
        context (str): Context describing the current cleaning operation.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while
                      reporting dataset cleaning, e.g.:
            * If one or both of datasets received are not Pandas DataFrame.
            * If one or both of datasets received are of incompatible type
              with operations used to calculate their lengths.
    """
    try:
        # Calculate original dataset size
        initial_df_len = len(original_df)

        debug(f"(Before {context}) Dataset length: {initial_df_len}")

        # Calculate final dataset size
        final_df_len = len(new_df)

        debug(f"(After {context}) Dataset length: {final_df_len}")

        debug(f"{initial_df_len - final_df_len} rows removed from dataset")
    except (AttributeError, TypeError) as e:
        msg = "Failed to report dataset cleaning"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
