import torch

from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """
    Enable Monte Carlo (MC) Dropout for inference.

    This function sets all dropout layers of the given model
    to training mode, allowing stochastic dropout to be applied
    during inference.

    Args:
        model (torch.nn.Module): The PyTorch model in which to
                                 enable MC Dropout.

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

    # Set a flag indicating MC Dropout is active
    model.mc_dropout = True

    info("MC Dropout enabled for model")
