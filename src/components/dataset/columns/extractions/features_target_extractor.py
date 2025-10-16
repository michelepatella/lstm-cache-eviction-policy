from typing import List, Tuple

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


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
        RuntimeError: If an error occurs while extracting features and target
                      from dataset columns e.g.:
                        * If column list is empty.
    """
    debug(f"Dataset columns to extract features and target from: {columns}")

    try:
        # Extract features and target
        features = columns[:-1]
        target = columns[-1]

        debug(
            f"Features: {features}, target: {target}, extracted from dataset columns"
        )
    except IndexError as e:
        msg = "Failed to extract features and target from dataset columns"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Features and target extracted from dataset columns")

    return features, target
