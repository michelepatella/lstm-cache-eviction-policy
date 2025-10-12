import torch
from fastapi import HTTPException, status


def select_device(device_type: str) -> torch.device:
    """
    Select and return the computation device.

    This function selects a computation device to be
    used by API computations — according to device type
    received — returning it.

    Args:
        device_type (str): The type of device to use.

    Returns:
        torch.device: The PyTorch device object corresponding
                      to the requested device type.

    Raises:
        HTTPException: If an error occurs during device selection, e.g.:
            * If the device type is invalid or not recognized.
            * If the device cannot be used due to runtime constraints.
    """
    try:
        # Instantiate device based on
        # device type passed
        device = torch.device(device_type)

        return device
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid device type '{device_type}'",
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Device '{device_type}' cannot be used",
        ) from e
