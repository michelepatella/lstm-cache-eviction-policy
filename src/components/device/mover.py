from typing import Union

import torch

from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def move_to_device(
    obj: Union[torch.nn.Module, torch.Tensor], device: torch.device
) -> Union[torch.nn.Module, torch.Tensor]:
    """
    Move an object to the specified device.

    This function moves the provided object (a PyTorch model or a Tensor)
    to the requested device and returns it.

    Args:
        obj (Union[torch.nn.Module, torch.Tensor]): Model/tensor to move to device.
        device (torch.device): Target device.

    Returns:
        Union[torch.nn.Module, torch.Tensor]: The object moved to the specified device.

    Raises:
        RuntimeError: If moving the object to the specified device fails:
            * Device type is invalid or not recognized (TypeError).
            * Object cannot be moved due to runtime constraints, such as insufficient
              memory or incompatible object type (RuntimeError).
    """
    try:
        # Move object to device
        obj = obj.to(device)

        info(f"{obj} moved to {device}")

        return obj
    except (TypeError, RuntimeError) as e:
        msg = "Failed to move object to specified device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
