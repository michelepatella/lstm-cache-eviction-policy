import torch

from pipeline.utils.logs.levels.error_logger import error


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
        # Instantiate device based on
        # device type passed
        device = torch.device(device_type)

        return device
    except (TypeError, RuntimeError) as e:
        msg = "Failed to select device"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
