"""scorer_service_caller.py

Module containing the logic to call the gRPC Scorer Service.

This module is responsible for communicating with the Scorer Service.
It handles the serialization of model output data (predictions and variances)
into the gRPC format, manages the gRPC channel connection, executes the
remote procedure call, and processes the response.

Functions:
    call_scorer_service(
        outputs: list[list[float]],
        variances: list[list[float]],
        api_config: APIConfig,
    ) -> np.array:
        Initiates the gRPC call to the Scorer Service to calculate key scores.
"""

import grpc
import numpy as np
from fastapi import HTTPException, status

import api.services.scorer.scorer_service_pb2 as pb2
import api.services.scorer.scorer_service_pb2_grpc as pb2_grpc
from api.config.pydantic.api_config import APIConfig
from api.const import SCORER_SERVICE_CHANNEL
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def call_scorer_service(
    outputs: list[list[float]],
    variances: list[list[float]],
    api_config: APIConfig,
) -> np.array:
    """Initiates the gRPC call to the Scorer Service to calculate key scores.

    This function opens a channel to the Scorer Service, serializes the
    model outputs, variances, and configuration parameters into the
    appropriate gRPC request format, and calls the remote gRPC method.

    Args:
        outputs (list[list[float]]): A list of model output sequences (logits)
                                     to be evaluated.
        variances (list[list[float]]): A list of variance sequences corresponding
                                       to the model outputs.
        api_config (APIConfig): API configuration object.

    Returns:
        np.array: An array containing the calculated scores for each key.

    Raises:
        HTTPException: If a gRPC communication error occurs, converted into a 500
                       Internal Server Error.
    """
    try:
        with grpc.insecure_channel(SCORER_SERVICE_CHANNEL) as ch:
            debug(
                "Scorer service call started",
                extra={
                    "outputs_num": len(outputs),
                    "variances_num": len(variances),
                    "conf_level": api_config.kwargs.conf_level.value,
                    "conf_weight": api_config.kwargs.conf_weight.value,
                    "context": "Scorer service call",
                },
            )

            # Create stub, build request for the service,
            # and call it retrieving the response
            stub = pb2_grpc.ScorerServiceStub(ch)
            request = pb2.ScorerServiceRequest(
                outputs=[f for sublist in outputs for f in sublist],
                variances=[v for sublist in variances for v in sublist],
                conf_level=api_config.kwargs.conf_level.value,
                conf_weight=api_config.kwargs.conf_weight.value,
            )
            response = stub.Score(request)

            debug(
                "Scorer service call completed",
                extra={
                    "key_scores_num": len(np.array(response.key_scores)),
                    "context": "Scorer service call",
                },
            )

            return np.array(response.key_scores)

    except grpc.RpcError as e:
        error(
            "Scorer service call failed",
            extra={
                "exception": str(e),
                "outputs_num": len(outputs),
                "variances_num": len(variances),
                "conf_level": api_config.kwargs.conf_level.value,
                "conf_weight": api_config.kwargs.conf_weight.value,
                "context": "Scorer service call",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
