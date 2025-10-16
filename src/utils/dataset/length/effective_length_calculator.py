import pandas as pd

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def calculate_effective_dataset_length(df: pd.DataFrame, seq_len: int) -> int:
    """
    Calculate effective dataset length.

    This function calculates and returns the effective dataset
    length, with respect to the sequence length provided.

    Args:
        df (pd.DataFrame): Dataset to calculate effective length for.
        seq_len (int): Length of sequences to extract.

    Returns:
        int: Number of available sequences.

    Raises:
        RuntimeError: If an error occurs while calculating effective
                      dataset length e.g.:
                        * If calculated length is negative.
                        * If dataset is invalid.
    """
    try:
        debug(
            f"Sequence length to calculate effective dataset length: {seq_len}"
        )

        # Calculate dataset length
        dataset_length = len(df) - seq_len
        debug(f"Calculated dataset length: {dataset_length}")

        # Check whether dataset length is negative
        if dataset_length < 0:
            msg = "Calculated effective dataset length is negative"
            error("%s", msg)
            raise RuntimeError(msg)

        info("Dataset length calculation completed")

        return dataset_length
    except (TypeError, AttributeError, ValueError) as e:
        msg = "Failed to calculate effective dataset length"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
