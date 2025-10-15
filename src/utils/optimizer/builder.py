from typing import Any, Dict

import torch
from torch.optim import Optimizer

from pipeline.const import ADAM_OPTIMIZER, ADAMW_OPTIMIZER, SGD_OPTIMIZER
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info

# Define optimizer mapping
OPTIMIZER_MAP: Dict[str, type[Optimizer]] = {
    ADAM_OPTIMIZER: torch.optim.Adam,
    ADAMW_OPTIMIZER: torch.optim.AdamW,
    SGD_OPTIMIZER: torch.optim.SGD,
}


def build_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    **optimizer_kwargs: Any,
) -> Optimizer:
    """
    Build an optimizer for the given model.

    This function creates an optimizer based on the provided type,
    with parameters received as arguments.

    Args:
        model (torch.nn.Module): The model for which to create the optimizer.
        optimizer_type (str): Optimizer type to be instantiated.
        **optimizer_kwargs (Any): Parameters for the optimizer.

    Returns:
        Optimizer: Initialized optimizer.

    Raises:
        RuntimeError: If an error occurs while initializing optimizer e.g.:
            * Model parameters are not valid for optimizer.
    """
    try:
        debug(f"Optimizer type to be built: {optimizer_type}")
        debug(f"Optimizer parameters: {optimizer_kwargs}")

        # Retrieve optimizer class from mapping
        # according to provided type
        optimizer_cls = OPTIMIZER_MAP.get(optimizer_type)

        # Instantiate optimizer
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        info("Optimizer built")

        return optimizer
    except (TypeError, ValueError) as e:
        msg = "Failed to build optimizer"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
