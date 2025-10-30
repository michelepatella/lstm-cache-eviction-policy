import requests
import torch
from box import Box
from fastapi import HTTPException, status

from api.config.kwargs.APIKwargs import APIKwargs
from api.const import (
    PREDICTOR_SERVICE_DEVICE_TYPE_PARAM_NAME,
    PREDICTOR_SERVICE_LAST_ACCESSES_PARAM_NAME,
    PREDICTOR_SERVICE_MC_DROPOUT_SAMPLES_PARAM_NAME,
    PREDICTOR_SERVICE_PARAMS,
    PREDICTOR_SERVICE_RETURN_OUTPUTS_NAME,
    PREDICTOR_SERVICE_RETURN_VARIANCES_NAME,
    PREDICTOR_SERVICE_ROLLOUT_HORIZON_PARAM_NAME,
    PREDICTOR_SERVICE_TIME_STEP_INCREMENT_PARAM_NAME,
    PREDICTOR_SERVICE_URL,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def call_predictor_service(
    last_accesses: list[tuple[float, int]],
    api_kwargs: APIKwargs,
    api_config: Box,
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
        api_kwargs (APIKwargs): API kwargs.
        api_config (Box): API configuration.

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
        params[PREDICTOR_SERVICE_LAST_ACCESSES_PARAM_NAME] = last_accesses
        params[PREDICTOR_SERVICE_DEVICE_TYPE_PARAM_NAME] = (
            api_config.hardware.device_type
        )
        params[PREDICTOR_SERVICE_ROLLOUT_HORIZON_PARAM_NAME] = (
            api_kwargs.rollout_horizon
        )
        params[PREDICTOR_SERVICE_MC_DROPOUT_SAMPLES_PARAM_NAME] = (
            api_kwargs.mc_dropout_samples
        )
        params[PREDICTOR_SERVICE_TIME_STEP_INCREMENT_PARAM_NAME] = (
            api_kwargs.time_step_increment
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
            PREDICTOR_SERVICE_URL,
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
