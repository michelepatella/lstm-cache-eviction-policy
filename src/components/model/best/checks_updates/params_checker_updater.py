import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def check_update_best_model_params(
    curr_avg_loss: float,
    best_avg_loss: float | None,
    curr_model_params: dict[str, int | float | bool],
    best_model_params: dict[str, int | float | bool],
) -> tuple[float, dict[str, int | float | bool]]:
    """Check and update the best model parameters based on the average loss.

    This function compares the current average loss with the best one found
    so far. If the new loss improves upon the previous best, both the best
    average loss and corresponding parameters are updated.

    Args:
        curr_avg_loss (float): Average loss of the current iteration.
        best_avg_loss (float | None): Best average loss found so far
                                      (None if not set yet).
        curr_model_params (dict[str, int | float | bool]):
            Current model parameters.
        best_model_params (dict[str, int | float | bool]):
            Best model parameters found so far.

    Returns:
        tuple[float, dict[str, int | float | bool]]:
            - best_avg_loss: Best average loss (updated or not).
            - best_params: The model parameters corresponding to the
                           best average loss (updated or not).

    Raises:
        RuntimeError: If checking and updating the best model parameters fails:
            * Comparison between current and best average loss fails due to
              invalid types (TypeError).
    """
    try:
        debug(
            "Best model parameters checking/updating started",
            extra={
                "loss_avg_current": None if np.isinf(curr_avg_loss) or np.isnan(curr_avg_loss) else float(curr_avg_loss),
                "loss_avg_best": None if np.isinf(best_avg_loss) or np.isnan(best_avg_loss) else float(best_avg_loss),
                "model_params_current": curr_model_params,
                "context": "Best model parameters checking/updating",
            },
        )

        # Update the best model parameters if
        # improvement in loss is found (or best average loss
        # not still set)
        if best_avg_loss is None or curr_avg_loss < best_avg_loss:
            best_avg_loss = curr_avg_loss
            best_model_params = curr_model_params

        debug(
            "Best model parameters checking/updating completed",
            extra={
                "loss_avg_current": None if np.isinf(curr_avg_loss) or np.isnan(curr_avg_loss) else float(curr_avg_loss),
                "loss_avg_best": None if np.isinf(best_avg_loss) or np.isnan(best_avg_loss) else float(best_avg_loss),
                "model_best_updated": curr_avg_loss
                < (
                    best_avg_loss
                    if best_avg_loss is not None
                    else float("inf")
                ),
                "context": "Best model parameters checking/updating",
            },
        )

        return best_avg_loss, best_model_params
    except TypeError as e:
        msg = "Best model parameters checking/updating failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "loss_avg_current": None if np.isinf(curr_avg_loss) or np.isnan(curr_avg_loss) else float(curr_avg_loss),
                "loss_avg_best": None if np.isinf(best_avg_loss) or np.isnan(best_avg_loss) else float(best_avg_loss),
                "context": "Best model parameters checking/updating",
            },
        )
        raise RuntimeError(msg) from e
