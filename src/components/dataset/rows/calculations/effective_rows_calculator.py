import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_effective_dataset_rows(df: pd.DataFrame, seq_len: int) -> int:
    """
    Calculate effective dataset rows.

    This function calculates and returns the effective dataset
    rows, with respect to the sequence length provided.

    Args:
        df (pd.DataFrame): Dataset to calculate effective length for.
        seq_len (int): Length of sequences to extract.

    Returns:
        int: Number of available sequences.

    Raises:
        RuntimeError: If dataset effective rows calculation fails:
            * Invalid arguments or data types (TypeError, ValueError).
            * Dataset is not a DataFrame (AttributeError).
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

        return dataset_length
    except (TypeError, AttributeError, ValueError) as e:
        msg = "Failed to calculate effective dataset length"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
