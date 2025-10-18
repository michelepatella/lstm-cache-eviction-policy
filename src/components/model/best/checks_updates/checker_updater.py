from typing import Dict, Tuple, Optional

import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.model.state_dict.copier import (
    copy_model_state_dict,
)


def check_update_best_model(
    curr_avg_loss: float, best_avg_loss: float, model: torch.nn.Module
) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
    """
    Update the best model weights if the current average loss improves.

    This function checks if the provided average loss is lower than
    the current best average loss. If it is, the model's state dictionary
    is copied and returned along with the updated best average loss.

    Args:
        curr_avg_loss (float): Current epoch average loss.
        best_avg_loss (float): Best average loss observed so far.
        model (torch.nn.Module): PyTorch model to copy if improvement is found.

    Returns:
        Tuple[float, Optional[Dict[str, torch.Tensor]]]:
            - best_avg_loss: Best average loss (updated or not).
            - best_model_weights: Best model weights. None if no improvement is found.

    Raises:
        RuntimeError: If checking and updating the best model fails:
            * Comparison between average loss and best average loss fails due to
              invalid types (TypeError).
    """
    debug(f"Current average loss: {curr_avg_loss}")
    debug(f"Best average loss: {best_avg_loss}")

    try:
        # Check for an average loss improvement
        if curr_avg_loss < best_avg_loss:
            # Update both best average loss
            # and model weights
            best_avg_loss = curr_avg_loss
            best_model_weights = copy_model_state_dict(model)

            info(
                f"Model check and update completed (New best avg loss: {best_avg_loss})"
            )

            return best_avg_loss, best_model_weights
    except TypeError as e:
        msg = "Failed to check and update the best model"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Model check and update completed (No improvement)")

    return best_avg_loss, None
