"""features_target_extractor.py

Utility module for extracting features and target from dataset columns.

This module provides the `extract_features_target_from_dataset_columns`
function, which separates features and target based on the target column
index.

Functions:
    extract_features_target_from_dataset_columns(
        columns: list[str],
        target_column_idx: int = DATASET_COLUMNS.index(DATASET_COLUMN_REQUEST_NAME)
    ) -> tuple[list[str], str]
        Extracts features (all columns before target) and the target column
        from a list of dataset columns.
"""

from components.const import DATASET_PROCESSED_COLUMNS
from components.logs.levels.error_logger import error
from const import DATASET_COLUMN_REQUEST_NAME


def extract_features_target_from_dataset_columns(
    columns: list[str],
    target_column_idx: int = DATASET_PROCESSED_COLUMNS.index(
        DATASET_COLUMN_REQUEST_NAME,
    ),
) -> tuple[list[str], str]:
    """Extract features and target from dataset columns.

    This function extracts both features and target from provided dataset
    columns, assuming the last column is the target while all the other ones
    are features.

    Args:
        columns (list[str]): List of dataset columns to extract features
                             and target for.
        target_column_idx (int): The index of the target column in the
                                 dataset.

    Returns:
        tuple[list[str], str]: List of features and target extracted.

    Raises:
        RuntimeError: If extracting features or target fails:
            * Columns list is empty (IndexError).
    """
    try:
        # Extract features and target
        features = columns[:target_column_idx]
        target = columns[target_column_idx]

        return features, target
    except IndexError as e:
        msg = "Features and target extraction from dataset columns failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "columns_provided": columns,
                "num_columns": len(columns) if columns else 0,
                "target_column_idx": target_column_idx,
                "context": "Features and target extraction from dataset columns",
            },
        )
        raise RuntimeError(msg) from e
