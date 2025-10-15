from torch import nn
import torch
from utils.device.mover import move_to_device
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.criterion.class_weights.calculator import calculate_class_weight


def build_criterion(
    targets: torch.Tensor, num_classes: int, device: torch.device
) -> nn.CrossEntropyLoss:
    """
    Build a PyTorch criterion.

    This function computes class weights based on the target labels,
    moves the weights to the specified device, and returns a
    PyTorch criterion configured with these weights.

    Args:
        targets (torch.Tensor): Targets for computing class weights.
        num_classes (int): Number of classes.
        device (torch.device): Device to move the criterion weights onto.

    Returns:
        nn.CrossEntropyLoss: Criterion with class weights.

    Raises:
        RuntimeError: If an error occurs while building criterion e.g.:
            * Device or tensor issues.
            * If class weights cannot be converted to a tensor of
              correct type.
    """
    debug(f"Number of targets for criterion: {len(targets)}")
    debug(f"Number of classes for criterion: {num_classes}")
    debug(f"Device for criterion: {device}")

    # Compute class weights
    class_weights = calculate_class_weight(targets, num_classes)

    try:
        # Move class weights as tensor to
        # the specified device
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        weight_tensor = move_to_device(weight_tensor, device)

        # Build criterion with computed weights
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        info(f"Criterion built")

        return criterion
    except (RuntimeError, TypeError) as e:
        msg = "Failed to build criterion"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
