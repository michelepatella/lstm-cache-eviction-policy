"""survival_uncertainty_calculator.py

This module implements the logic for calculating item scores using a hybrid
approach that combines survival analysis concepts with prediction uncertainty.

The scoring mechanism evaluates how likely an item is to be "survived" (not
accessed) over a rollout horizon while penalizing items with high prediction
variance. This combined score is used to rank items for cache eviction
decisions.

Functions:
    calculate_survival_uncertainty_scores(
        outputs: list[np.ndarray],
        variances: list[np.ndarray],
        conf_level: float,
        conf_weight: float,
    ) -> np.ndarray:
        Calculates normalized scores for cache items based on model outputs
        and confidence intervals.
"""

import numpy as np
from sklearn.preprocessing import minmax_scale

from components.const import AUTOREGRESSIVE_ROLLOUT_DIM
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.math.confidence_interval_calculator import (
    calculate_confidence_interval,
)


def calculate_survival_uncertainty_scores(
    outputs: list[np.ndarray],
    variances: list[np.ndarray],
    conf_level: float,
    conf_weight: float,
) -> np.ndarray:
    """Calculates normalized scores combining survival delay and uncertainty.

    This function processes model logits and variances across a rollout horizon
    to determine the eviction priority. It computes confidence interval widths
    to quantify uncertainty and uses a cumulative product of shifted logits to
    estimate a survival-based access delay. The final scores are min-max scaled
    for consistency.

    Args:
        outputs (list[np.ndarray]): List of predicted logit tensors from the
                                    model rollout.
        variances (list[np.ndarray]): List of variance tensors corresponding to
                                      the model outputs.
        conf_level (float): The confidence level used to calculate the width of
                            the confidence intervals.
        conf_weight (float): A weighting factor applied to the uncertainty
                             contribution of the total score.

    Returns:
        np.ndarray: Normalized scores (0 to 1) for each key, where higher values
                    indicate lower eviction priority.

    Raises:
        RuntimeError: If calculation fails due to:
            * Input array shape mismatch (ValueError).
            * Non-numeric or missing data (TypeError).
            * Degenerate inputs causing issues in normalization (ValueError).
    """
    try:
        debug(
            "Survival-uncertainty item scores calculation started",
            extra={
                "conf_level": conf_level,
                "conf_weight": conf_weight,
                "context": "Survival-uncertainty item scores calculation",
            },
        )

        # Stack outputs (logits) and shift them
        logits = np.stack(outputs)
        shifted_logits = logits - logits.min(
            AUTOREGRESSIVE_ROLLOUT_DIM,
            keepdims=True,
        )

        # Calculate confidence intervals
        lower_ci, upper_ci = calculate_confidence_interval(
            outputs,
            variances,
            conf_level,
        )

        # Calculate uncertainty as the width
        # of confidence intervals
        uncertainty = upper_ci.cpu().numpy() - lower_ci.cpu().numpy()

        # Convert the mean uncertainty into a positive
        # score contribution where higher uncertainty
        # decreases this term
        uncertainty_score = conf_weight * np.exp(
            uncertainty.mean(AUTOREGRESSIVE_ROLLOUT_DIM),
        )

        # Survival-based access delay score over
        # the rollout horizon
        survival_score = np.cumprod(
            np.log1p(shifted_logits),
            AUTOREGRESSIVE_ROLLOUT_DIM,
        ).sum(AUTOREGRESSIVE_ROLLOUT_DIM) / len(logits)

        # Calculate and normalize final key scores
        # as the combination of survival and uncertainty scores
        key_scores = minmax_scale(survival_score + uncertainty_score)

        debug(
            "Survival-uncertainty item scores calculation completed",
            extra={
                "key_scores_num": len(key_scores),
                "conf_level": conf_level,
                "conf_weight": conf_weight,
                "context": "Survival-uncertainty item scores calculation",
            },
        )

        return key_scores
    except (ValueError, TypeError) as e:
        msg = "Survival-uncertainty item scores calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "conf_level": conf_level,
                "conf_weight": conf_weight,
                "context": "Survival-uncertainty item scores calculation",
            },
        )
        raise RuntimeError(msg) from e
