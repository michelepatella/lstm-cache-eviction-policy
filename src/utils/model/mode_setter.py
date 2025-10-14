import torch

from pipeline.const import MC_DROPOUT_MODEL_MODE, TRAINING_MODEL_MODE
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def set_model_mode(
    model: torch.nn.Module,
    mode: str,
    mc_dropout_flag: str = None
) -> None:
    """
    Set the mode of a PyTorch model.

    This function, given a model and a mode, set
    the model in that mode. Optionally, enables MC
    dropout by setting a provided flag.

    Args:
        model (torch.nn.Module): Model to modify.
        mode (str): Desired mode.
        mc_dropout_flag (str): Attribute to set as flag on the model.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while setting model mode e.g.:
            * If the mode is invalid.
            * If the mode is incompatible.
    """
    try:
        if mode == MC_DROPOUT_MODEL_MODE:
            # Set all model layers to evaluation mode
            model.eval()

            # Set only dropout layer(s) to
            # training mode
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

            debug("MC Dropout enabled")

            # Set MC Dropout flag
            if mc_dropout_flag is not None and hasattr(model, mc_dropout_flag):
                setattr(model, mc_dropout_flag, True)
                debug(f"MC Dropout flag set")
        elif mode == TRAINING_MODEL_MODE:
            # Set model to training mode
            model.train()
            debug("Model set to training mode")
        else:
            # Set model to evaluation mode
            model.eval()
            debug("Model set to evaluation mode")
    except (AttributeError, ValueError, TypeError) as e:
        msg = "Failed to set model mode"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Model mode setting completed")