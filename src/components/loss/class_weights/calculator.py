import numpy as np
from sklearn.utils import compute_class_weight
from torch import Tensor

from const import CLASS_WEIGHT_TYPE
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def calculate_class_weight(
    targets: Tensor,
    num_classes: int,
    weight_type: str = CLASS_WEIGHT_TYPE,
) -> np.ndarray:
    """
    Compute balanced class weight for targets.

    This function ensures that classes which appear less frequently
    are assigned higher weights, useful for imbalanced classification.

    Args:
        targets (Tensor): Target labels.
        num_classes (int): Number of classes.
        weight_type (str): The weight type to apply.

    Returns:
        np.ndarray: Array of class weights.

    Raises:
        RuntimeError: If an error occurs during computation, e.g.:
            * Targets cannot be converted to a NumPy array.
            * Computation of class weight fails.
    """
    try:
        # Convert targets to NumPy array
        targets_array = targets.cpu().numpy()

        debug(
            f"Targets converted to NumPy array "
            f"with shape: {targets_array.shape}"
        )

        # Identify present classes in targets
        present_classes = np.unique(targets_array)

        debug(f"Present classes in targets: {present_classes}")

        # Compute class weight for present classes
        weights = compute_class_weight(
            class_weight=weight_type,
            classes=present_classes,
            y=targets_array,
        )

        debug(f"Weights for present classes: {weights}")

        # Initialize full class weights array
        class_weights = np.ones(num_classes, dtype=np.float32)

        # Update weights for appearing classes
        for cls, weight in zip(present_classes, weights):
            class_weights[cls] = weight

        debug(f"Class weight array: {class_weights}")

        info("Class weight calculation completed")

        return class_weights
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        msg = "Failed to calculate class weight"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
