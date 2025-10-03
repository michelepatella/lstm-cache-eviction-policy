import pandas as pd

from const import (
    MISSING_VALUES_REMOVAL_DROPNA_AXIS,
    MISSING_VALUES_REMOVAL_DROPNA_HOW,
)
from pipeline.data_preprocessing.visualization.report.dataset_removal_reporter import (
    report_dataset_removal,
)
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def remove_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove missing values from dataset.

    This function removes rows with missing values
    from dataset, returning a new clean dataset.

    Parameters:
        df (pd.DataFrame): Dataset to remove missing
                           values from.

    Returns:
        pd.DataFrame: Dataset without missing values.

    Raises:
        RuntimeError: If dataframe to be cleaned is
                      not a pd.DataFrame.
    """
    try:
        # Remove rows with missing values
        new_df = df.dropna(
            axis=MISSING_VALUES_REMOVAL_DROPNA_AXIS,
            how=MISSING_VALUES_REMOVAL_DROPNA_HOW,
        )

        # For missing values report
        report_dataset_removal(df, new_df, "Missing values removal")

        info("Missing values removal completed")

        return new_df
    except AttributeError as e:
        msg = "Failed to remove missing values from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
