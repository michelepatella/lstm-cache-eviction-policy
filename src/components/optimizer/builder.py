from typing import Any, Dict

import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from src.const import (
    OPTIMIZER_ADAM_NAME,
    OPTIMIZER_ADAMW_NAME,
    OPTIMIZER_SGD_NAME,
)

# Map each optimizer type to
# its PyTorch instance
OPTIMIZER_MAP: Dict[str, type[torch.optim.Optimizer]] = {
    OPTIMIZER_ADAM_NAME: torch.optim.Adam,
    OPTIMIZER_ADAMW_NAME: torch.optim.AdamW,
    OPTIMIZER_SGD_NAME: torch.optim.SGD,
}


def build_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Build an optimizer for the given model.

    This function creates an optimizer for the provided model,
    based on the requested type, with parameters received as arguments.

    Args:
        model (torch.nn.Module): PyTorch model for which to create the optimizer.
        optimizer_type (str): Optimizer type to be instantiated.
        **optimizer_kwargs (Any): Parameters for the optimizer.

    Returns:
        torch.optim.Optimizer: Built optimizer.

    Raises:
        RuntimeError: If optimizer building fails:
            * Model parameters are invalid or incompatible with optimizer
              (TypeError, ValueError).
    """
    try:
        debug(f"Optimizer type to be built: {optimizer_type}")
        debug(f"Optimizer parameters: {optimizer_kwargs}")

        # Retrieve requested optimizer
        # instance from mapping
        optimizer_cls = OPTIMIZER_MAP.get(optimizer_type)

        # Instantiate optimizer
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        debug(f"Optimizer built: {optimizer_type}")

        return optimizer
    except (TypeError, ValueError) as e:
        msg = "Failed to build optimizer"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
