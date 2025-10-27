import numpy as np
import torch
from fastapi import HTTPException, status
from sklearn.preprocessing import minmax_scale
from torch import softmax

from components.const import TENSOR_OUTPUTS_BATCH_DIM
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def calculate_key_scores(
    outputs: list[torch.Tensor],
    lower_ci: torch.Tensor,
    upper_ci: torch.Tensor,
    prob_weight: float,
    conf_weight: float,
) -> tuple[dict[int, float], np.ndarray, np.ndarray]:
    """Calculate key scores based on predicted probabilities
    and confidence intervals.

    This function computes a score (in [0,1]) for each key
    using the probability of being used at each predicted
    time step and the confidence of the prediction.

    Args:
        outputs (list[torch.Tensor]): List of model outputs per time step.
        lower_ci (torch.Tensor): Tensor of lower bounds of confidence
                                 intervals per step.
        upper_ci (torch.Tensor): Tensor of upper bounds of confidence
                                 intervals per step.
        prob_weight (float): Weight of the probability in the final score.
        conf_weight (float): Weight of the confidence in the final score.

    Returns:
        tuple[dict[int, float], np.ndarray, np.ndarray]:
            - dict[int, float]: Dictionary mapping each key to its
                                normalized score.
            - np.ndarray: Probability matrix.
            - np.ndarray: Confidence matrix.

    Raises:
        HTTPException: If an error occurs during scores calculation:
            * Shape mismatch or invalid type in outputs/confidence intervals
              (TypeError).
            * Division by zero or invalid normalization (ValueError).
    """
    try:
        debug(
            "Key score calculation started",
            extra={
                "outputs_num": len(outputs),
                "steps_num": len(outputs),
                "keys_num": len(outputs[0]),
                "lower_ci_shape": tuple(lower_ci.shape),
                "upper_ci_shape": tuple(upper_ci.shape),
                "prob_weight": prob_weight,
                "conf_weight": conf_weight,
                "context": "Key score calculation",
            },
        )

        # Build probability and confidence matrices
        # having keys as rows and time steps as columns
        # so that each cell (i,j) is filled by the
        # probability/prediction confidence of key i
        # at time step j
        prob_matrix = np.stack(
            [
                softmax(o, dim=TENSOR_OUTPUTS_BATCH_DIM).cpu().numpy()
                for o in outputs
            ],
        )
        conf_matrix = 1 / (upper_ci - lower_ci)

        # Normalize confidence matrix forcing
        # values in [0,1]
        conf_matrix = minmax_scale(conf_matrix)

        # For each key, compute a balanced score based on
        # the probability of being used over all predicted
        # time steps and corresponding prediction confidence
        scores_array = prob_weight * prob_matrix + conf_weight * conf_matrix
        key_scores = scores_array.sum(axis=0)

        # Normalize key scores forcing values
        # in [0,1]
        key_scores = minmax_scale(key_scores)

        debug(
            "Key score calculation completed",
            extra={
                "key_scores": key_scores,
                "prob_matrix_shape": prob_matrix.shape,
                "conf_matrix_shape": conf_matrix.shape,
                "context": "Key score calculation",
            },
        )

        return (
            {k: float(score) for k, score in enumerate(key_scores)},
            prob_matrix,
            conf_matrix,
        )
    except (TypeError, ValueError) as e:
        error(
            "Key score calculation failed",
            extra={
                "exception": str(e),
                "outputs_num": len(outputs),
                "steps_num": len(outputs),
                "keys_num": len(outputs[0]),
                "lower_ci_shape": tuple(lower_ci.shape),
                "upper_ci_shape": tuple(upper_ci.shape),
                "prob_weight": prob_weight,
                "conf_weight": conf_weight,
                "context": "Key score calculation",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
