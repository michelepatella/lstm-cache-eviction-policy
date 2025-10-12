import torch
from torch.optim import Optimizer

from config.classes.Config import Config
from const import ADAM_OPTIMIZER, ADAMW_OPTIMIZER
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def build_optimizer(
    model: torch.nn.Module, learning_rate: float, optimizer_type: str, config: Config
) -> Optimizer:
    """
    Build an optimizer for the given model.

    This function creates an optimizer based on the provided type.
    Supports 'adam', 'adamw', and 'sgd' optimizers.

    Args:
        model (torch.nn.Module): The model for which to create the optimizer.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_type (str): Optimizer type to be instantiated.
        config (Config): Configuration object.

    Returns:
        Optimizer: Initialized optimizer.

    Raises:
        RuntimeError: If an error occurs while initializing optimizer e.g.:
            * Model parameters are not valid for optimizer.
            * Invalid numeric parameters.
    """
    try:
        debug(f"Optimizer type: {optimizer_type}")
        debug(f"Learning rate for optimizer: {learning_rate}")

        # Retrieve model parameters
        model_params = model.parameters()

        if optimizer_type == ADAM_OPTIMIZER:
            # Instantiate Adam optimizer
            optimizer = torch.optim.Adam(
                model_params,
                lr=learning_rate,
            )

            debug("Adam optimizer selected")
        elif optimizer_type == ADAMW_OPTIMIZER:
            # Retrieve weight decay
            weight_decay = config.training.optimizer.params.weight_decay

            # Instantiate AdamW optimizer
            optimizer = torch.optim.AdamW(
                model_params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            debug(f"AdamW optimizer selected, weight decay: {weight_decay}")
        else:
            # Retrieve momentum
            momentum = config.training.optimizer.params.momentum

            # Instantiate SGD
            optimizer = torch.optim.SGD(
                model_params,
                lr=learning_rate,
                momentum=momentum,
            )

            debug(f"SGD optimizer selected, momentum: {momentum}")
    except (TypeError, ValueError) as e:
        msg = "Failed to build optimizer"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Optimizer built")

    return optimizer
