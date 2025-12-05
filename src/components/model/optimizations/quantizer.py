"""quantizer.py

Utility module for the dynamical quantization of PyTorch models.

This module provides the `quantize_model` function, which applies dynamic
quantization to a PyTorch model to reduce memory footprint and potentially
improve inference speed on CPU. The function allows specifying target
modules, quantization dtype, and the quantization engine.

Functions:
    quantize_model(
        model: torch.nn.Module,
        dtype: str,
        qengine: str,
        target_modules: tuple[
            type[torch.nn.Module],
            ...,
        ] = (torch.nn.Linear,)
    ) -> torch.nn.Module
        Apply dynamic quantization to the given PyTorch model.
"""

import torch
from torch.ao.quantization import default_dynamic_qconfig, quantize_dynamic

from components.device.mover import move_to_device
from components.device.selector import select_device
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from src.const import RESOURCES_DEVICE_CPU_NAME


def quantize_model(
    model: torch.nn.Module,
    dtype: str,
    qengine: str,
    target_modules: tuple[
        type[torch.nn.Module],
        ...,
    ] = (torch.nn.Linear,),
) -> torch.nn.Module:
    """Apply dynamic quantization to a PyTorch model.

    This function applies dynamic quantization to the specified
    modules of a PyTorch model to reduce its memory footprint.

    Args:
        model (torch.nn.Module): The PyTorch model to quantize.
        dtype (str): Quantization data type.
        qengine (str): Quantization engine to use.
        target_modules (tuple[type[torch.nn.Module], ...]):
            Layer types to apply quantization on.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        RuntimeError: If model quantization fails:
            * Invalid module types or unsupported dtype (TypeError, ValueError).
            * Unsupported operations on the current device or backend
              (NotImplementedError, AttributeError).
            * Internal quantization errors from PyTorch (RuntimeError, OSError).
    """
    try:
        debug(
            "Model quantization started",
            extra={
                "model_type": type(model).__name__,
                "target_modules": [m.__name__ for m in target_modules],
                "dtype": str(dtype),
                "context": "Model quantization",
            },
        )

        # Move model to CPU before
        # running quantization
        device = select_device(RESOURCES_DEVICE_CPU_NAME)
        model = move_to_device(model, device)

        # Set quantization engine
        torch.backends.quantized.engine = qengine

        # Prepare qconfig_spec as a dict for
        # dynamic quantization
        qconfig_spec = dict.fromkeys(target_modules, default_dynamic_qconfig)

        # Get dtype form string
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec=qconfig_spec,
            dtype=dtype,
        )

        debug(
            "Model quantization completed",
            extra={
                "model_type": type(model).__name__,
                "target_modules": [m.__name__ for m in target_modules],
                "dtype": str(dtype),
                "engine": torch.backends.quantized.engine,
                "device": str(device),
                "context": "Model quantization",
            },
        )

        return quantized_model
    except (
        ValueError,
        TypeError,
        AttributeError,
        NotImplementedError,
        OSError,
        RuntimeError,
    ) as e:
        msg = "Model quantization failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "model_type": type(model).__name__,
                "target_modules": [m.__name__ for m in target_modules],
                "dtype": str(dtype),
                "context": "Model quantization",
            },
        )
        raise RuntimeError(msg) from e
