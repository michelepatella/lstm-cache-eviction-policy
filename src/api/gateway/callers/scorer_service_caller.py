import numpy as np
import requests
import torch
from box import Box
from fastapi import HTTPException, status

from api.config.kwargs.APIKwargs import APIKwargs
from api.const import (
    SCORER_SERVICE_PARAM_CONF_WEIGHT_NAME,
    SCORER_SERVICE_PARAM_CONFIDENCE_LEVEL_NAME,
    SCORER_SERVICE_PARAM_OUTPUTS_NAME,
    SCORER_SERVICE_PARAM_PROB_WEIGHT_NAME,
    SCORER_SERVICE_PARAM_VARIANCES_NAME,
    SCORER_SERVICE_PARAMS,
    SCORER_SERVICE_RETURN_CONF_MATRIX_NAME,
    SCORER_SERVICE_RETURN_KEY_SCORES_NAME,
    SCORER_SERVICE_RETURN_PROB_MATRIX_NAME,
    SCORER_SERVICE_URL,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def call_scorer_service(
    outputs: list[torch.Tensor],
    variances: list[torch.Tensor],
    api_kwargs: APIKwargs,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Call scorer service.

    This function sends predicted outputs and variances
    to the scorer service, which calculates key scores
    based on probabilities and prediction confidence. It
    returns the key scores along with probability and
    confidence matrices.

    Args:
        outputs (list[torch.Tensor]): Predicted outputs from the
                                      predictor service.
        variances (list[torch.Tensor]): Corresponding variances for
                                        predicted outputs.
        api_kwargs (APIKwargs): API kwargs.

    Returns:
        tuple[list[float], np.ndarray, np.ndarray]:
            - key_scores: Mapping from key index to normalized score.
            - prob_matrix: Probability matrix used for scoring.
            - conf_matrix: Confidence matrix used for scoring.

    Raises:
        HTTPException: If scorer service call fails:
            * Network or connection issues (requests.RequestException).
            * Response parsing fails (ValueError, KeyError).
            * Returned data does not contain expected fields (KeyError).
    """
    try:
        # Prepare parameters for scorer service
        params = Box(SCORER_SERVICE_PARAMS)
        params[SCORER_SERVICE_PARAM_OUTPUTS_NAME] = outputs
        params[SCORER_SERVICE_PARAM_VARIANCES_NAME] = variances
        params[SCORER_SERVICE_PARAM_CONFIDENCE_LEVEL_NAME] = (
            api_kwargs.confidence_level
        )
        params[SCORER_SERVICE_PARAM_PROB_WEIGHT_NAME] = api_kwargs.prob_weight
        params[SCORER_SERVICE_PARAM_CONF_WEIGHT_NAME] = api_kwargs.conf_weight

        debug(
            "Scorer service call started",
            extra={
                "params": params.to_dict(),
                "context": "Scorer service",
            },
        )

        # Call scorer service and box the response
        response = requests.post(
            SCORER_SERVICE_URL,
            json=params.to_dict(),
        )
        data = Box(response.json())

        # Extract service responses
        key_scores = data.get(SCORER_SERVICE_RETURN_KEY_SCORES_NAME)
        prob_matrix = data.get(SCORER_SERVICE_RETURN_PROB_MATRIX_NAME)
        conf_matrix = data.get(SCORER_SERVICE_RETURN_CONF_MATRIX_NAME)

        debug(
            "Scorer service call completed",
            extra={
                "key_scores_num": len(key_scores) if key_scores else 0,
                "context": "Scorer service",
            },
        )

        return key_scores, prob_matrix, conf_matrix
    except (requests.RequestException, ValueError, KeyError) as e:
        error(
            "Scorer service call failed",
            extra={
                "exception": str(e),
                "outputs_num": len(outputs),
                "variances_num": len(variances),
                "context": "Scorer service",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
