import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def select_device(device_type: str) -> torch.device:
    """
    Select and return the requested computation device.

    This function selects and returns a computation device, according
    to device type requested.

    Args:
        device_type (str): The type of device to be selected.

    Returns:
        torch.device: The PyTorch device object corresponding to the
                      requested device type.

    Raises:
        RuntimeError: If selecting the requested device fails:
            * Device type is invalid or not recognized (TypeError).
            * Device cannot be used due to runtime constraints (RuntimeError).
    """
    debug(f"Device type to be selected: {device_type}")

    try:
        # Instantiate device based on
        # device type requested
        device = torch.device(device_type)
    except (TypeError, RuntimeError) as e:
        msg = "Failed to select device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Device '{device_type}' selected")

    return device
