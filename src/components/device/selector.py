"""selector.py

Utility module for selecting a PyTorch computation device.

This module provides the `select_device` function, which returns a
PyTorch `torch.device` object according to the requested device type.

Functions:
    select_device(device_type: str) -> torch.device
        Select and return the requested computation device.
"""

import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def select_device(device_type: str) -> torch.device:
    """Select and return the requested computation device.

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
    try:
        debug(
            "Device selection started",
            extra={
                "device_type_requested": device_type,
                "context": "Device selection",
            },
        )

        # Instantiate device based on
        # device type requested
        device = torch.device(device_type)

        debug(
            "Device selection completed",
            extra={
                "device_type_selected": str(device),
                "context": "Device selection",
            },
        )

        return device
    except (TypeError, RuntimeError) as e:
        msg = "Device selection failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "device_type_requested": device_type,
                "context": "Device selection",
            },
        )
        raise RuntimeError(msg) from e
