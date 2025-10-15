import copy
from typing import Dict

import torch

from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def copy_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Copy state dictionary of a PyTorch model.

    This function creates a copy of the given model's
    state dictionary. It ensures that changes to the copied
    dictionary do not affect the original model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        Dict[str, torch.Tensor]: Copy of the model's state dictionary.

    Raises:
        RuntimeError: If an error occurs while copying the model
                      state dictionary e.g.:
                        * If the model is corrupted or not
                          fully initialized.
    """
    try:
        # Copy model state dictionary
        model_state_dict = copy.deepcopy(model.state_dict())

        info("Model state dictionary copied")

        return model_state_dict
    except TypeError as e:
        msg = "Failed to copy model state dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
