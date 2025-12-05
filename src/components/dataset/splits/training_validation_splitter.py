"""training_validation_splitter.py

Utility module for splitting datasets into training and validation sets.

This module provides the `split_training_validation_sets` function, which
splits a given dataset into training and validation portions. It can
use provided indices or compute them based on a validation split percentage,
returning sets for each portion.

Functions:
    split_training_validation_sets(
        training_set: AccessLogsDataset,
        validation_split: float | None = None,
        training_idx: np.ndarray | None = None,
        validation_idx: np.ndarray | None = None
    ) -> tuple[AccessLogsDataset, AccessLogsDataset]
        Splits a dataset into training and validation sets.
"""

import numpy as np

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.index.calculator import (
    calculate_dataset_split_index,
)
from components.logs.levels.error_logger import error
from pipeline.const import DATASET_RESET_INDEX_DROP


def split_training_validation_sets(
    training_set: AccessLogsDataset,
    validation_split: float | None = None,
    training_idx: np.ndarray | None = None,
    validation_idx: np.ndarray | None = None,
) -> tuple[AccessLogsDataset, AccessLogsDataset]:
    """Split the dataset into training and validation sets.

    This function either uses provided indices or calculates them based on
    the validation percentage, and returns the training and validation sets.

    Args:
        training_set (AccessLogsDataset): The dataset to split.
        validation_split (float | None): Validation percentage to split dataset.
        training_idx (np.ndarray | None): Training set index.
        validation_idx (np.ndarray | None): Validation set index.

    Returns:
        tuple[AccessLogsDataset, AccessLogsDataset]:
            - final_training_set: The set containing the training
                                  portion of the dataset.
            - final_validation_set: The set containing the validation
                                    portion of the dataset.

    Raises:
        RuntimeError: If training/validation set split fails:
            * Invalid dataset type or index type (TypeError).
            * Invalid split percentage or calculation (ValueError).
            * Dataset object missing attributes (AttributeError).
            * Index out of bounds (IndexError).
            * Undefined variable or other name errors (NameError).
    """
    try:
        # If both or one of the two index is not
        # provided, calculate them
        if training_idx is None or validation_idx is None:
            # Calculate total training set size
            total_training_size = len(training_set)

            # Calculate training and validation sizes
            training_size = calculate_dataset_split_index(
                total_training_size,
                1 - validation_split,
            )
            validation_size = calculate_dataset_split_index(
                total_training_size,
                validation_split,
            )

            # Generate training and validation indices
            training_idx = list(range(training_size))
            validation_idx = list(
                range(training_size, training_size + validation_size),
            )

        # Create sets both for
        # training and validation sets
        training_df = training_set.data.iloc[training_idx].reset_index(
            drop=DATASET_RESET_INDEX_DROP,
        )
        validation_df = training_set.data.iloc[validation_idx].reset_index(
            drop=DATASET_RESET_INDEX_DROP,
        )

        # Build AccessLogsDataset both for
        # training and testing set
        final_training_set = AccessLogsDataset(
            None,
            None,
            None,
            parent=training_set,
            data=training_df,
        )
        final_validation_set = AccessLogsDataset(
            None,
            None,
            None,
            parent=training_set,
            data=validation_df,
        )

        return final_training_set, final_validation_set
    except (TypeError, ValueError, AttributeError, IndexError, NameError) as e:
        msg = "Training and validation set splitting failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "validation_split": validation_split,
                "training_idx_provided": training_idx is not None,
                "validation_idx_provided": validation_idx is not None,
                "dataset_length": (
                    len(training_set)
                    if hasattr(training_set, "__len__")
                    else None
                ),
                "context": "Training and validation set splitting",
            },
        )
        raise RuntimeError(msg) from e
