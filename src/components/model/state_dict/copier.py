import copy
from typing import Dict

import torch

from components.logs.levels.error_logger import error


def copy_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Copy state dictionary of a PyTorch model.

    This function creates a copy of the given model's state dictionary.
    It ensures that changes to the copied dictionary do not affect the
    original model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        Dict[str, torch.Tensor]: Copy of the model's state dictionary.

    Raises:
        RuntimeError: If copying the model state dictionary fails:
            * Deepcopy fails due to unsupported types or invalid tensors
              (TypeError).
    """
    try:
        # Copy model state dictionary
        model_state_dict = copy.deepcopy(model.state_dict())

        return model_state_dict
    except TypeError as e:
        msg = "Model state dictionary copying failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "model_type": type(model).__name__,
                "params_num": len(list(model.state_dict().keys())),
                "context": "Model state dictionary copying",
            },
        )
        raise RuntimeError(msg) from e
