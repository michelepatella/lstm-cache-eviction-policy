import pandas as pd

from lstm_eviction_policy.data_preprocessing.utils.dataset_cleaning_reporter import (
    report_dataset_cleaning,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    error,
    info,
)


def remove_duplicates(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """
    Remove duplicates from dataset.

    This function removes rows with duplicated values
    from a given subset of dataset column(s),
    returning a new clean dataset.

    Parameters:
        df (pd.DataFrame): Dataset to remove duplicates from.
        subset (list[str]): Column(s) of dataset to remove
                            duplicates from.

    Returns:
        pd.DataFrame: Dataset without duplicated values.

    Raises:
        AttributeError: If dataframe to be cleaned is
                        not a pd.DataFrame.
    """
    try:
        # Remove duplicates from dataset
        new_df = df.drop_duplicates(subset=subset)

        # For duplicates removal report
        report_dataset_cleaning(df, new_df, "Duplicates removal")

        info("Duplicates removal completed")

        return new_df
    except AttributeError as e:
        msg = "Failed to remove duplicates from dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
