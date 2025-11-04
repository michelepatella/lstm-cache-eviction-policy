"""setter.py

Utility module to set the mode of a PyTorch model.

This module provides the `set_model_mode` function, which allows switching
a PyTorch model between standard training, evaluation, or MC Dropout modes.
It optionally sets a flag on the model to indicate MC Dropout is enabled.

Functions:
    set_model_mode(
        model: torch.nn.Module,
        mode: str,
        mc_dropout_flag_name: str = None,
        mc_dropout_flag_value: Any = MC_DROPOUT_ENABLED
    ) -> None
        Set the mode of a PyTorch model, optionally enabling MC Dropout.
"""

from typing import Any

import torch

from components.const import (
    MC_DROPOUT_ENABLED,
    MODEL_MC_DROPOUT_MODE,
    MODEL_TRAIN_MODE,
)
from components.logs.levels.error_logger import error


def set_model_mode(
    model: torch.nn.Module,
    mode: str,
    mc_dropout_flag_name: str = None,
    mc_dropout_flag_value: Any = MC_DROPOUT_ENABLED,
) -> None:
    """Set the mode of a PyTorch model.

    This function, given a model and a mode, set the model in that mode.
    Optionally, enables MC dropout by setting a provided flag to the model.

    Args:
        model (torch.nn.Module): PyTorch model for which mode is to set.
        mode (str): Desired mode.
        mc_dropout_flag_name (str): Attribute to set as flag on the model
                                    (only if requested mode is MC Dropout).
        mc_dropout_flag_value (Any): Value corresponding to the attribute
                                     to set as flag on the model (only if
                                     requested mode is MC Dropout).

    Returns:
        None

    Raises:
        RuntimeError: If setting the model mode fails:
            * Attribute assignment to the model fails due to invalid
              attribute name or type (AttributeError, TypeError).
            * Invalid mode value causes failure (ValueError).
    """
    try:
        # MC Dropout mode
        if mode == MODEL_MC_DROPOUT_MODE:
            # Set all model layers to
            # evaluation mode
            model.eval()

            # Set only dropout layer(s) to
            # training mode
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

            # Set MC Dropout flag to the model
            if mc_dropout_flag_name is not None:
                setattr(model, mc_dropout_flag_name, mc_dropout_flag_value)

        # Training mode
        elif mode == MODEL_TRAIN_MODE:
            # Set model to training mode
            model.train()

        # Fallback: Evaluation mode
        else:
            # Set model to evaluation mode
            model.eval()
    except (AttributeError, ValueError, TypeError) as e:
        msg = "Model mode setting failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "mode": mode,
                "mc_dropout_flag_name": mc_dropout_flag_name,
                "mc_dropout_flag_value": mc_dropout_flag_value,
                "model_type": type(model).__name__,
                "context": "Model mode setting",
            },
        )
        raise RuntimeError(msg) from e
