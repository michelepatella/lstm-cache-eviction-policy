import torch

from utils.logs.levels.error_logger import error


def move_to_device(
    obj: torch.nn.Module | torch.Tensor, device: torch.device
) -> torch.nn.Module | torch.Tensor:
    """
    Move an object to the specified device.

    This function moves the provided object (a PyTorch model
    or a Tensor) to the requested device and returns it.

    Args:
        obj (torch.nn.Module | torch.Tensor): Model or tensor to move.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module | torch.Tensor: The object on the specified device.

    Raises:
        RuntimeError: If an error occurs while moving the object
                      to the specified device, e.g.:
            * If the device type is invalid or not recognized.
            * If the object cannot be moved due to runtime constraints.
    """
    try:
        # Move object to device
        obj = obj.to(device)

        return obj
    except (TypeError, RuntimeError) as e:
        msg = "Failed to move the object to the specified device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
