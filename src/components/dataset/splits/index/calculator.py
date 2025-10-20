from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_dataset_split_index(dataset_len: int, split_perc: float) -> int:
    """
    Calculate split index for a dataset.

    This function calculates the split index for a given dataset,
    according to the provided percentage.

    Args:
        dataset_len (int): Length of the dataset.
        split_perc (float): Fraction to split.

    Returns:
        int: Dataset split index.

    Raises:
        RuntimeError: If split index calculation fails:
            * Invalid dataset length or split percentage type (TypeError).
            * Invalid numeric value (ValueError).
    """
    try:
        debug(f"Dataset length to calculate split index for: {dataset_len}")
        debug(f"Dataset splitting percentage: {split_perc}")

        # Calculate split index
        split_idx = int(dataset_len * split_perc)
        debug(f"Split index calculated: {split_idx}")

        return split_idx
    except (TypeError, ValueError) as e:
        msg = "Failed to calculate dataset split index"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
