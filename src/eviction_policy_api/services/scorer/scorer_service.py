import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.math.confidence_interval_calculator import (
    calculate_confidence_interval,
)
from eviction_policy_api.const import (
    SCORER_SERVICE_ENDPOINT,
    SCORER_SERVICE_RETURN_CONF_MATRIX_NAME,
    SCORER_SERVICE_RETURN_KEY_SCORES_NAME,
    SCORER_SERVICE_RETURN_PROB_MATRIX_NAME,
)
from eviction_policy_api.services.scorer.scores.calculator import calculate_key_scores

app = FastAPI()


@app.post(SCORER_SERVICE_ENDPOINT)
def scorer_service(
    outputs: list[torch.Tensor],
    variances: list[torch.Tensor],
    confidence_level: float,
    prob_weight: float,
    conf_weight: float,
) -> dict[str, dict[int, float] | np.ndarray]:
    """Compute key scores based on model outputs
    and their confidence intervals.

    This endpoint calculates confidence intervals for
    each output using its variance and the specified
    confidence level. Then, it computes a key score
    for each key considering both the probability of
    being used and the confidence of the prediction.

    Args:
        outputs (list[torch.Tensor]): List of model outputs per predicted
                                      step.
        variances (list[torch.Tensor]): List of variances corresponding to
                                        each output.
        confidence_level (float): Confidence level to compute intervals.
        prob_weight (float): Weight of probability contribution to the score.
        conf_weight (float): Weight of confidence contribution to the score.

    Returns:
        dict[str, dict[int, float] | np.ndarray]:
            - key_scores: Normalized key scores.
            - prob_matrix: Probability matrix used for key scores calculation.
            - conf_matrix: Confidence matrix used for key scores calculation.

    Raises:
        HTTPException: If the score service fails:
            * Confidence intervals calculation fails (RuntimeError).
    """
    try:
        debug(
            "Scorer service started",
            extra={
                "outputs_num": len(outputs),
                "variances_num": len(variances),
                "confidence_level": confidence_level,
                "prob_weight": prob_weight,
                "conf_weight": conf_weight,
                "context": "Scorer service",
            },
        )

        # Calculate confidence intervals given the
        # outputs and their corresponding variances,
        # according to the specified confidence level
        lower_ci, upper_ci = calculate_confidence_interval(
            outputs,
            variances,
            confidence_level,
        )

        # For each key, calculate a score based
        # on the probability of being used
        # at each predicted future step and
        # the confidence of that model prediction
        key_scores, prob_matrix, conf_matrix = calculate_key_scores(
            outputs,
            lower_ci,
            upper_ci,
            prob_weight,
            conf_weight,
        )

        debug(
            "Scorer service completed",
            extra={
                "key_scores": key_scores,
                "prob_matrix_shape": prob_matrix.shape,
                "conf_matrix_shape": conf_matrix.shape,
                "context": "Scorer service",
            },
        )

        return {
            SCORER_SERVICE_RETURN_KEY_SCORES_NAME: key_scores,
            SCORER_SERVICE_RETURN_PROB_MATRIX_NAME: prob_matrix,
            SCORER_SERVICE_RETURN_CONF_MATRIX_NAME: conf_matrix,
        }
    except RuntimeError as e:
        error(
            "Scorer service failed",
            extra={
                "exception": str(e),
                "outputs_num": len(outputs),
                "variances_num": len(variances),
                "confidence_level": confidence_level,
                "prob_weight": prob_weight,
                "conf_weight": conf_weight,
                "context": "Scorer service",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
