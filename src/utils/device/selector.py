import torch

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def select_device(device_type: str) -> torch.device:
    """
    Select and return the computation device.

    This function selects a computation device —
    according to device type received — returning it.

    Args:
        device_type (str): The type of device to use.

    Returns:
        torch.device: The PyTorch device object corresponding
                      to the requested device type.

    Raises:
        RuntimeError: If an error occurs during device selection, e.g.:
            * If the device type is invalid or not recognized.
            * If the device cannot be used due to runtime constraints.
    """
    try:
        debug(f"Device type to be instantiated: {device_type}")

        # Instantiate device based on
        # device type passed
        device = torch.device(device_type)

        info(f"Device selected")

        return device
    except (TypeError, RuntimeError) as e:
        msg = "Failed to select device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
