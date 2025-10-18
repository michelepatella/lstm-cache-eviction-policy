from typing import Optional

import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import (
    MC_DROPOUT_MODEL_MODE,
    TRAINING_MODEL_MODE,
    MC_DROPOUT_ENABLED,
)


def set_model_mode(
    model: torch.nn.Module,
    mode: str,
    mc_dropout_flag_name: Optional[str] = None,
    mc_dropout_flag_value: Optional[bool] = MC_DROPOUT_ENABLED,
) -> None:
    """
    Set the mode of a PyTorch model.

    This function, given a model and a mode, set the model in that mode.
    Optionally, enables MC dropout by setting a provided flag to the model.

    Args:
        model (torch.nn.Module): PyTorch model for which mode is to set.
        mode (str): Desired mode.
        mc_dropout_flag_name (Optional[str]): Attribute to set as flag on the model
                                              (only if requested mode is MC Dropout).
        mc_dropout_flag_value (Optional[bool]): Value corresponding to the attribute
                                                to set as flag on the model (only if
                                                requested mode is MC Dropout).

    Returns:
        None

    Raises:
        RuntimeError: If setting the model mode fails:
            * Attribute assignment to the model fails due to invalid attribute name
              or type (AttributeError, TypeError).
            * Invalid mode value causes failure (ValueError).
    """
    try:
        # MC Dropout mode
        if mode == MC_DROPOUT_MODEL_MODE:
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
                debug(f"MC Dropout flag set to model")

        # Training mode
        elif mode == TRAINING_MODEL_MODE:
            # Set model to training mode
            model.train()

        # Fallback: Evaluation mode
        else:
            # Set model to evaluation mode
            model.eval()

        info(f"Model mode set: {mode}")
    except (AttributeError, ValueError, TypeError) as e:
        msg = "Failed to set model mode"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
