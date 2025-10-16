from typing import Dict, Tuple

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def check_update_best_model_params(
    avg_loss: float,
    best_avg_loss: float | None,
    curr_params: Dict[str, int | float | bool],
    best_params: Dict[str, int | float | bool],
) -> Tuple[float, Dict[str, int | float | bool]]:
    """
    Check and update the best parameters
    based on the average loss.

    This function compares the current average loss
    with the best one found so far. If the new loss
    improves upon the previous best, both the best loss and
    corresponding parameters are updated.

    Args:
        avg_loss (float): Average loss of the current iteration.
        best_avg_loss (float | None): Best average loss found so far,
                                      or None if not set yet.
        curr_params (Dict[str, int | float | bool]): Current parameter set
                                                     being evaluated.
        best_params (Dict[str, int | float | bool]): Best parameter set found
                                                     so far.

    Returns:
        Tuple[float, Dict[str, int | float | bool]]:
            - best_avg_loss: The updated best average loss after comparison.
            - best_params: The parameter set corresponding to the best average loss.

    Raises:
        RuntimeError: If an error occurs while checking and
                      updating best params e.g.:
                        * If average loss or the best one
                          (when provided) are not numeric.
    """
    debug(f"Current avg loss: {avg_loss}")
    debug(f"Best avg loss so far: {best_avg_loss}")

    try:
        # Update the best parameters if
        # improvement is found
        if best_avg_loss is None or avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_params = curr_params

            info("Updated best parameters and average loss")
            debug(f"New best parameters: {best_params}")
            debug(f"New best avg loss: {best_avg_loss}")
        else:
            info("No improvement in average loss, best parameters unchanged")
    except ValueError as e:
        msg = "Failed to check and update best parameters"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info("Best parameters check completed")

    return best_avg_loss, best_params
