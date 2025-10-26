import torch
from torch import nn

from components.device.mover import move_to_device
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.loss.class_weight.calculator import calculate_class_weight


def build_loss(
    targets: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """Build a PyTorch criterion.

    This function computes class weight based on the target labels,
    moves the weight to the specified device, and returns a
    PyTorch criterion configured with these weight.

    Args:
        targets (torch.Tensor): Targets for computing class weight.
        num_classes (int): Number of classes.
        device (torch.device): Device to move the criterion weight onto.

    Returns:
        nn.CrossEntropyLoss: Criterion with class weight.

    Raises:
        RuntimeError: If building the loss fails:
            * Conversion of class weight to tensor fails due to invalid
              type or shape (TypeError).
            * Moving the class weight tensor to device fails due to device
              or tensor issues (RuntimeError).
            * Creating CrossEntropyLoss with the class weight fails due to
              invalid tensor or shape (RuntimeError, TypeError).
    """
    try:
        debug(
            "Loss building started",
            extra={
                "targets_num": (
                    len(targets) if hasattr(targets, "__len__") else None
                ),
                "classes_num": num_classes,
                "device": str(device),
                "context": "Loss building",
            },
        )

        # Compute class weight
        class_weight = calculate_class_weight(targets, num_classes)

        # Move class weight as tensor to
        # the specified device
        class_weight_tensor = torch.tensor(class_weight, dtype=torch.float32)
        class_weight_tensor = move_to_device(class_weight_tensor, device)

        # Build criterion with computed weight
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

        debug(
            "Loss building completed",
            extra={
                "targets_num": (
                    len(targets) if hasattr(targets, "__len__") else None
                ),
                "classes_num": num_classes,
                "class_weight_shape": tuple(class_weight_tensor.shape),
                "device": str(device),
                "context": "Loss building",
            },
        )

        return criterion
    except (RuntimeError, TypeError) as e:
        msg = "Loss building failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "targets_num": (
                    len(targets) if hasattr(targets, "__len__") else None
                ),
                "classes_num": num_classes,
                "targets_dtype": str(getattr(targets, "dtype", None)),
                "targets_shape": tuple(getattr(targets, "shape", ())),
                "device": str(device),
                "context": "Loss building",
            },
        )
        raise RuntimeError(msg) from e
