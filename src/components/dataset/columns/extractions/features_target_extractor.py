from typing import List, Tuple

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def extract_features_target_from_dataset_columns(
    columns: List[str],
) -> Tuple[List[str], str]:
    """
    Extract features and target from dataset columns.

    This function extracts both features and target from provided dataset
    columns, assuming the last column is the target while all the other ones
    are features.

    Args:
        columns (List[str]): List of dataset columns to extract features
                             and target for.

    Returns:
        Tuple[List[str], str]: List of features and target extracted.

    Raises:
        RuntimeError: If extracting features or target fails:
            * Columns list is empty (IndexError).
    """
    try:
        debug(
            f"Dataset columns to extract features and target from: {columns}"
        )

        # Extract features and target
        features = columns[:-1]
        target = columns[-1]

        debug(
            f"Features ({features}) and target ({target}) extracted"
            f" from dataset columns"
        )

        return features, target
    except IndexError as e:
        msg = "Failed to extract features and target from dataset columns"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
