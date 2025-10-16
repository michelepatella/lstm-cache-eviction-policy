from typing import Dict, Tuple

import torch

from components.logs.levels.info_logger import info
from components.model.state_dict.copier import (
    copy_model_state_dict,
)


def check_update_best_model(
    avg_loss: float, best_avg_loss: float, model: torch.nn.Module
) -> Tuple[float, Dict[str, torch.Tensor] | None]:
    """
    Update the best model weights if the current average loss improves.

    This function checks if the provided average loss is lower than
    the current best average loss. If it is, the model's state dictionary
    is copied and returned along with the updated best average loss.

    Args:
        avg_loss (float): Current epoch average loss.
        best_avg_loss (float): Best average loss observed so far.
        model (torch.nn.Module): Model to copy if improvement is found.

    Returns:
        Dict[str, torch.Tensor] | None: Updated best average loss
                                        and new best model weights
                                        (or None if no improvement).
    """
    # Check for an average loss improvement
    if avg_loss < best_avg_loss:
        # Update both best average loss and
        # model weights
        best_avg_loss = avg_loss
        best_model_weights = copy_model_state_dict(model)

        info(f"Model updated (New best avg loss: {best_avg_loss})")

        return best_avg_loss, best_model_weights

    info("Model update completed (No improvement)")

    return best_avg_loss, None
