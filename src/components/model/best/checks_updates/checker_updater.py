import numpy as np
import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.model.state_dict.copier import (
    copy_model_state_dict,
)


def check_update_best_model(
    curr_avg_loss: float,
    best_avg_loss: float,
    model: torch.nn.Module,
) -> tuple[float, dict[str, torch.Tensor] | None]:
    """Update the best model weights if the current average loss improves.

    This function checks if the provided average loss is lower than
    the current best average loss. If it is, the model's state dictionary
    is copied and returned along with the updated best average loss.

    Args:
        curr_avg_loss (float): Current epoch average loss.
        best_avg_loss (float): Best average loss observed so far.
        model (torch.nn.Module): PyTorch model to copy if improvement is found.

    Returns:
        tuple[float, dict[str, torch.Tensor] | None]:
            - best_avg_loss: Best average loss (updated or not).
            - best_model_weights: Best model weights. None if no
                                  improvement is found.

    Raises:
        RuntimeError: If checking and updating the best model fails:
            * Comparison between average loss and best average loss
              fails due to invalid types (TypeError).
    """
    try:
        debug(
            "Best model checking/updating started",
            extra={
                "loss_avg_current": None
                if np.isinf(curr_avg_loss) or np.isnan(curr_avg_loss)
                else float(curr_avg_loss),
                "loss_avg_best": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
                "model_type": type(model).__name__,
                "context": "Best model checking/updating",
            },
        )

        # Check for an average loss improvement
        best_model_weights = None
        if curr_avg_loss < best_avg_loss:
            # Update both best average loss
            # and model weights
            best_avg_loss = curr_avg_loss
            best_model_weights = copy_model_state_dict(model)

        debug(
            "Best model checking/updating completed",
            extra={
                "loss_avg_current": None
                if np.isinf(curr_avg_loss) or np.isnan(curr_avg_loss)
                else float(curr_avg_loss),
                "loss_avg_best": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
                "model_best_updated": best_model_weights is not None,
                "model_type": type(model).__name__,
                "context": "Best model checking/updating",
            },
        )

        return best_avg_loss, best_model_weights
    except TypeError as e:
        msg = "Best model checking/updating failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "loss_avg_current": None
                if np.isinf(curr_avg_loss) or np.isnan(curr_avg_loss)
                else float(curr_avg_loss),
                "loss_avg_best": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
                "model_type": type(model).__name__,
                "context": "Best model checking/updating",
            },
        )
        raise RuntimeError(msg) from e
