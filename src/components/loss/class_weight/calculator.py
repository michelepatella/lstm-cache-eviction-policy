import numpy as np
from sklearn.utils import compute_class_weight
from sympy.printing.pytorch import torch

from components.const import CRITERION_CLASS_WEIGHT_TYPE
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_class_weight(
    targets: torch.Tensor,
    num_classes: int,
    weight_type: str = CRITERION_CLASS_WEIGHT_TYPE,
) -> np.ndarray:
    """Compute balanced class weight for targets.

    This function calculates class weight ensuring that classes
    which appear less frequently are assigned higher weights,
    useful for imbalanced classification.

    Args:
        targets (torch.Tensor): Target labels.
        num_classes (int): Number of classes.
        weight_type (str): The weight type to apply.

    Returns:
        np.ndarray: Array of class weight.

    Raises:
        RuntimeError: If class weight calculation fails:
            * Converting targets to NumPy array fails because targets is not a
              valid Tensor or has incompatible type
              (TypeError, AttributeError).
            * Computing unique classes or class weights fails due to invalid
              array or invalid inputs (ValueError, TypeError).
            * Assigning computed weights to full class weight array fails due
              to index issues (IndexError).
    """
    try:
        debug(
            "Class weight calculation started",
            extra={
                "classes_num": num_classes,
                "weight_type": weight_type,
                "targets_type": type(targets).__name__,
                "targets_shape": (
                    tuple(targets.shape)
                    if isinstance(targets, torch.Tensor)
                    else None
                ),
                "context": "Class weight calculation",
            },
        )

        # Convert targets to NumPy array
        targets_array = targets.cpu().numpy()

        # Identify present classes in targets
        present_classes = np.unique(targets_array)

        # Compute class weight for present classes
        weights = compute_class_weight(
            class_weight=weight_type,
            classes=present_classes,
            y=targets_array,
        )

        # Set weight for appearing classes
        class_weight = np.ones(num_classes, dtype=np.float32)
        for cls, weight in zip(present_classes, weights):
            class_weight[cls] = weight

        debug(
            "Class weight calculation completed",
            extra={
                "classes_present": present_classes.tolist(),
                "weights_computed": weights.tolist(),
                "class_weights": class_weight.tolist(),
                "context": "Class weight calculation",
            },
        )

        return class_weight
    except (TypeError, AttributeError, IndexError, ValueError) as e:
        msg = "Class weight calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "classes_num": num_classes,
                "weight_type": weight_type,
                "targets_type": type(targets).__name__,
                "targets_shape": (
                    tuple(targets.shape)
                    if isinstance(targets, torch.Tensor)
                    else None
                ),
                "context": "Class weight calculation",
            },
        )
        raise RuntimeError(msg) from e
