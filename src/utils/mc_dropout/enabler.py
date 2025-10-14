import torch

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def enable_mc_dropout(model: torch.nn.Module, mc_dropout_flag: str = None) -> None:
    """
    Enable Monte Carlo (MC) Dropout for inference.

    This function sets all dropout layers of the given model
    to training mode, allowing stochastic dropout to be applied
    during inference.

    Args:
        model (torch.nn.Module): The PyTorch model in which to
                                 enable MC Dropout.
        mc_dropout_flag (str): Name of the attribute
                               to set as flag on the model.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while enabling MC Dropout, e.g.:
            * If the received model is of type torch.nn.Module.
    """
    try:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                # Enable MC dropout by setting the
                # model to training mode during inference
                module.train()
    except AttributeError as e:
        msg = "Failed to enable MC dropout for model"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    # Set a flag indicating MC Dropout is active,
    # provided it has been passed and exists
    if mc_dropout_flag is not None and hasattr(model, mc_dropout_flag):
        setattr(model, mc_dropout_flag, True)
        debug("MC Dropout flag set")

    info("MC Dropout enabled for model")
