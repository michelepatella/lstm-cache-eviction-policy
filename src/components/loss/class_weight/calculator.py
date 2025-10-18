from typing import Optional

import numpy as np
from sklearn.utils import compute_class_weight
from sympy.printing.pytorch import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import CLASS_WEIGHT_TYPE


def calculate_class_weight(
    targets: torch.Tensor,
    num_classes: int,
    weight_type: Optional[str] = CLASS_WEIGHT_TYPE,
) -> np.ndarray:
    """
    Compute balanced class weight for targets.

    This function calculates class weight ensuring that classes
    which appear less frequently are assigned higher weights,
    useful for imbalanced classification.

    Args:
        targets (torch.Tensor): Target labels.
        num_classes (int): Number of classes.
        weight_type (Optional[str]): The weight type to apply.

    Returns:
        np.ndarray: Array of class weight.

    Raises:
        RuntimeError: If class weight calculation fails:
            * Converting targets to NumPy array fails because targets is not a
              valid Tensor or has incompatible type (TypeError, AttributeError).
            * Computing unique classes or class weights fails due to invalid array or
              invalid inputs (ValueError, TypeError).
            * Assigning computed weights to full class weight array fails due to index issues
              (IndexError).
    """
    try:
        # Convert targets to NumPy array
        targets_array = targets.cpu().numpy()
        debug(f"Targets array shape for class weight calculation: {targets_array.shape}")
    except (TypeError, AttributeError) as e:
        msg = "Failed to convert targets to NumPy array"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    try:
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
    except (ValueError, TypeError) as e:
        msg = "Failed to compute weights for present classes"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    try:
        # Set weight for appearing classes
        class_weight = np.ones(num_classes, dtype=np.float32)
        for cls, weight in zip(present_classes, weights):
            class_weight[cls] = weight
    except IndexError as e:
        msg = "Failed to set class weight"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Class weight calculated: {class_weight}")

    return class_weight
