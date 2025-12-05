import requests
import torch
from box import Box
from fastapi import HTTPException, status

from api.config.pydantic.api_config import APIConfig
from api.const import (
    PREDICTOR_SERVICE_FULL_URL,
    PREDICTOR_SERVICE_PARAM_LAST_ACCESSES_NAME,
    PREDICTOR_SERVICE_PARAM_MC_DROPOUT_SAMPLES_NAME,
    PREDICTOR_SERVICE_PARAM_ROLLOUT_HORIZON_NAME,
    PREDICTOR_SERVICE_PARAM_TIME_STEP_INCREMENT_NAME,
    PREDICTOR_SERVICE_PARAM_UNBIASED_VARIANCE_NAME,
    PREDICTOR_SERVICE_PARAMS,
    PREDICTOR_SERVICE_RETURN_OUTPUTS_NAME,
    PREDICTOR_SERVICE_RETURN_VARIANCES_NAME,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def call_predictor_service(
    last_accesses: list[tuple[float, int]],
    api_config: APIConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Call predictor service.

    This function calls the predictor service to perform
    autoregressive rollout and obtain predicted outputs
    along with their variances.

    Args:
        last_accesses (list[tuple[float, int]]): List of tuples
                                                 representing the
                                                 last access time
                                                 and corresponding key.
        api_config (APIConfig): API configuration object.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]:
            - outputs: Predicted outputs.
            - variances: Variances corresponding to the
                         predicted outputs.

    Raises:
        HTTPException: If predictor service call fails:
            * Network or connection issues (requests.RequestException).
            * Response parsing fails (ValueError, KeyError).
            * Returned data does not contain expected fields (KeyError).
    """
    try:
        # Prepare parameters for predictor service
        params = Box(PREDICTOR_SERVICE_PARAMS)
        params[PREDICTOR_SERVICE_PARAM_LAST_ACCESSES_NAME] = last_accesses
        params[PREDICTOR_SERVICE_PARAM_ROLLOUT_HORIZON_NAME] = (
            api_config.kwargs.rollout_horizon.value
        )
        params[PREDICTOR_SERVICE_PARAM_MC_DROPOUT_SAMPLES_NAME] = (
            api_config.kwargs.mc_dropout_samples.value
        )
        params[PREDICTOR_SERVICE_PARAM_TIME_STEP_INCREMENT_NAME] = (
            api_config.kwargs.time_step_increment.value
        )
        params[PREDICTOR_SERVICE_PARAM_UNBIASED_VARIANCE_NAME] = (
            api_config.kwargs.unbiased_variance.value
        )

        debug(
            "Predictor service call started",
            extra={
                "params": params.to_dict(),
                "context": "Predictor service",
            },
        )

        # Call predictor service and box the response
        response = requests.post(
            PREDICTOR_SERVICE_FULL_URL,
            json=params.to_dict(),
        )
        data = Box(response.json())

        # Extract service responses
        outputs = data.get(PREDICTOR_SERVICE_RETURN_OUTPUTS_NAME)
        variances = data.get(PREDICTOR_SERVICE_RETURN_VARIANCES_NAME)

        debug(
            "Predictor service call completed",
            extra={
                "outputs_num": len(outputs) if outputs else 0,
                "variances_num": len(variances) if variances else 0,
                "context": "Predictor service",
            },
        )

        return outputs, variances
    except (requests.RequestException, ValueError, KeyError) as e:
        error(
            "Predictor service call failed",
            extra={
                "exception": str(e),
                "last_accesses_num": len(last_accesses),
                "context": "Predictor service",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
