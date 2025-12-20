"""loader.py

Utility module for loading pre-trained PyTorch models.

This module provides the `load_model` function, which handles loading
a PyTorch model from disk, optionally setting a quantization engine,
and moving the model to a specified device.

Functions:
    load_model(
        path: str,
        device: torch.device = None,
        qengine: str = None,
    ) -> torch.nn.Module
        Loads a PyTorch model from the specified path, optionally moves it
        to the given device, and configures the quantization engine.
"""

import torch

from components.const import (
    MODEL_LOADING_WEIGHTS_ONLY,
)
from components.device.mover import move_to_device
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def load_model(
    path: str,
    device: torch.device = None,
    qengine: str = None,
) -> torch.nn.Module:
    """Load a PyTorch model.

    This function loads pre-trained PyTorch model from the
    specified path, moving it to the given device. Additionally,
    the function sets a quantization engine.

    Args:
        path (str): Path to the saved model.
        device (torch.device): Device to load the model to.
        qengine (str): Engine to set up for the model.

    Returns:
        torch.nn.Module: The loaded model moved to given device.

    Raises:
        RuntimeError: If loading the model fails:
            * Loading the model from file fails due to file issues,
              invalid path, or incompatible data (OSError, RuntimeError,
              TypeError).
    """
    try:
        debug(
            "Model loading started",
            extra={
                "path": str(path),
                "device": str(device) if device is not None else None,
                "context": "Model loading",
            },
        )

        # Setup engine for the quantized model
        if qengine is not None:
            torch.backends.quantized.engine = qengine

        # Load model
        model = torch.load(
            path,
            weights_only=MODEL_LOADING_WEIGHTS_ONLY,
        )  # nosec B614

        # Move model to device (optional)
        if device is not None:
            model = move_to_device(model, device)

        debug(
            "Model loading completed",
            extra={
                "path": str(path),
                "model_type": type(model).__name__,
                "device": str(device) if device is not None else None,
                "context": "Model loading",
            },
        )

        return model
    except (OSError, RuntimeError, TypeError) as e:
        msg = "Model loading failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": str(path),
                "device": str(device) if device is not None else None,
                "context": "Model loading",
            },
        )
        raise RuntimeError(msg) from e
