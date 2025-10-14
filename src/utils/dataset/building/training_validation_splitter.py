from typing import List, Tuple

from torch.utils.data import Subset

from pipeline.config.classes.Config import Config
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def split_training_validation(
    training_set: AccessLogsDataset,
    validation_split: float = None,
    training_idx: List[int] | None = None,
    validation_idx: List[int] | None = None,
) -> Tuple[Subset, Subset]:
    """
    Split the dataset into training and validation sets.

    This function either uses provided indices or calculates them based on
    the validation percentage, and returns PyTorch Subsets for training
    and validation.

    Args:
        training_set (AccessLogsDataset): The dataset to split.
        validation_split (float): Validation percentage to split dataset.
        training_idx (List[int] | None): Training set index.
        validation_idx (List[int] | None): Validation set index.

    Returns:
        Tuple[Subset, Subset]: Training and validation subsets.

    Raises:
        RuntimeError: If any error occurs during training set splitting, e.g.:
            * Indices calculation fails due to wrong types or values.
            * Provided indices are out of bounds.
    """
    try:
        # If both or one of the two index is not
        # provided, calculate them
        if training_idx is None or validation_idx is None:
            # Calculate total training set size
            total_training_size = len(training_set)

            debug(f"Total training set size: {total_training_size}")

            # Calculate training and validation sizes
            training_size = int((1.0 - validation_split) * total_training_size)
            validation_size = int(validation_split * total_training_size)

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
    except (TypeError, ValueError, AttributeError, IndexError, NameError) as e:
        msg = "Failed to split training and validation sets"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Training and validation sets split")

    return final_training_set, final_validation_set
