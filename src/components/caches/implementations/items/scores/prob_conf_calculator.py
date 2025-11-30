"""prob_conf_calculator.py

Module for calculating combined scores for cache items based on prediction
probability and confidence.

This module provides a function that processes raw model outputs (scores and variances)
and generates a final weighted score for each item. This score combines the item's
predicted probability (using softmax) and its confidence (derived from the width of
the confidence interval).

Functions:
     calculate_prob_conf_item_scores(
        outputs: list[list[float]],
        variances: list[list[float]],
        conf_level: float,
        prob_weight: float,
        conf_weight: float
    ) -> np.ndarray
        Calculates a final weighted score for each item.
"""

import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
from torch import softmax

from components.const import TENSOR_BATCH_DIM
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.math.confidence_interval_calculator import (
    calculate_confidence_interval,
)


def calculate_prob_conf_item_scores(
    outputs: list[list[float]],
    variances: list[list[float]],
    conf_level: float,
    prob_weight: float,
    conf_weight: float,
) -> np.ndarray:
    """Calculates a combined score for each item based on probability and
    confidence.

    This function first determines the confidence interval for the predicted
    outputs. It then calculates a probability matrix (using softmax) and a
    confidence matrix (based on the inverse width of the CI, normalized).
    Finally, it combines these matrices using specified weights to produce a
    single, normalized score for each item.

    Args:
        outputs (list[list[float]]): Model raw outputs (logits).
        variances (list[list[float]]): Corresponding variance/uncertainty values
                                       for the outputs.
        conf_level (float): The confidence level used for calculating the confidence
                            interval.
        prob_weight (float): The weight assigned to the probability component
                             when calculating the final score.
        conf_weight (float): The weight assigned to the confidence component
                             when calculating the final score.

    Returns:
        np.ndarray: An array of final, normalized scores for each item.

    Raises:
        RuntimeError: If calculation fails due to:
            * Input array shape mismatch (ValueError).
            * Non-numeric or missing data (TypeError).
            * Degenerate inputs causing issues in normalization (ValueError).
    """
    try:
        debug(
            "Prob-conf item scores calculation started",
            extra={
                "outputs_shape": f"{len(outputs)}x{len(outputs[0])}",
                "conf_level": conf_level,
                "prob_weight": prob_weight,
                "conf_weight": conf_weight,
                "context": "Prob-conf item scores calculation",
            },
        )

        # Calculate confidence intervals
        # given outputs and corresponding variances,
        # according to the provided confidence level
        lower_ci, upper_ci = calculate_confidence_interval(
            outputs,
            variances,
            conf_level,
        )

        # Build a probability matrix, where:
        # row = key
        # column = time step
        # and each cell is filled with corresponding probability
        prob_matrix = np.stack(
            [
                softmax(torch.tensor(o), dim=TENSOR_BATCH_DIM).cpu().numpy()
                for o in outputs
            ],
        )

        # Build a confidence matrix, where:
        # row = key
        # column = time step
        # and each cell is filled with corresponding confidence
        # (values in [0.0, 1.0])
        # Confidence is inverse of the CI width
        conf_matrix = 1 / (upper_ci - lower_ci)
        conf_matrix = minmax_scale(conf_matrix)

        # Calculate a score for each key (in [0.0, 1.0])
        # The key_scores are weighted by the sum of scores across all time steps
        key_scores = prob_matrix * (prob_weight + conf_weight * conf_matrix)
        key_scores = minmax_scale(key_scores.sum(axis=0))

        debug(
            "Prob-conf item scores calculation completed",
            extra={
                "final_scores_num": len(key_scores),
                "context": "Prob-conf item scores calculation",
            },
        )

        return key_scores
    except (ValueError, TypeError) as e:
        msg = "Prob-conf item scores calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "conf_level": conf_level,
                "prob_weight": prob_weight,
                "conf_weight": conf_weight,
                "context": "Prob-conf item scores calculation",
            },
        )
        raise RuntimeError(msg) from e
