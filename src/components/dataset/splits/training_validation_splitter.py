from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Subset

from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.index.calculator import (
    calculate_dataset_split_index,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def split_training_validation_sets(
    training_set: AccessLogsDataset,
    validation_split: Optional[float] = None,
    training_idx: Optional[np.ndarray] = None,
    validation_idx: Optional[np.ndarray] = None,
) -> Tuple[Subset, Subset]:
    """
    Split the dataset into training and validation sets.

    This function either uses provided indices or calculates them based on
    the validation percentage, and returns PyTorch Subsets for training
    and validation.

    Args:
        training_set (AccessLogsDataset): The dataset to split.
        validation_split (Optional[float]): Validation percentage to split dataset.
        training_idx (Optional[np.ndarray]): Training set index.
        validation_idx (Optional[np.ndarray]): Validation set index.

    Returns:
        Tuple[Subset, Subset]:
            - final_training_set: PyTorch Subset containing the training portion of
                                  the dataset.
            - final_validation_set: PyTorch Subset containing the validation portion
                                    of the dataset.

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
            debug(f"Total training set size: {total_training_size}")

            # Calculate training and validation sizes
            training_size = calculate_dataset_split_index(
                total_training_size, 1 - validation_split
            )
            validation_size = calculate_dataset_split_index(
                total_training_size, validation_split
            )
            debug(
                f"Calculated training size: {training_size},"
                f" validation size: {validation_size}"
            )

            # Generate training and validation indices
            training_idx = list(range(training_size))
            validation_idx = list(
                range(training_size, training_size + validation_size)
            )
            debug(
                f"Calculated training index: {training_idx},"
                f" validation index: {validation_idx}"
            )

        # Create Subset objects both for
        # training and validation sets
        final_training_set = Subset(training_set, training_idx)
        final_validation_set = Subset(training_set, validation_idx)

        debug("Training and validation sets split")

        return final_training_set, final_validation_set
    except (TypeError, ValueError, AttributeError, IndexError, NameError) as e:
        msg = "Failed to split training and validation sets"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
