"""predictor_service_caller.py

Module containing the logic to call the gRPC Predictor Service.

This module is responsible for communicating with the Predictor Service.
It handles the serialization of features, keys, and hyperparameters into the
gRPC request format, manages the gRPC channel connection, executes the remote
procedure call, and processes the output predictions and variances.

Functions:
    call_predictor_service(
        features: list[float],
        keys_seq: list[int],
        features_shape: list[int],
        keys_shape: list[int],
        api_config: APIConfig,
    ) -> tuple[list[list[float]], list[list[float]]]:
        Initiates the gRPC call to the Predictor Service for model inference.
"""

import grpc
from fastapi import HTTPException, status

import api.services.predictor.predictor_service_pb2 as pb2
import api.services.predictor.predictor_service_pb2_grpc as pb2_grpc
from api.config.pydantic.api_config import APIConfig
from api.const import PREDICTOR_SERVICE_CHANNEL
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def call_predictor_service(
    features: list[float],
    keys_seq: list[int],
    features_shape: list[int],
    keys_shape: list[int],
    api_config: APIConfig,
) -> tuple[list[list[float]], list[list[float]]]:
    """Initiates the gRPC call to the Predictor Service for model inference.

    This function opens a channel to the Predictor Service, serializes the
    pre-processed feature and key data along with inference configuration,
    and calls the remote gRPC prediction method.

    Args:
        features (list[float]): List of feature values.
        keys_seq (list[int]): List of accessed keys in sequence.
        features_shape (list[int]): Shape of the feature tensor.
        keys_shape (list[int]): Shape of the keys sequence tensor.
        api_config (APIConfig): API configuration object.

    Returns:
        tuple[list[list[float]], list[list[float]]]:
            - outputs: List of lists containing the predicted values.
            - variances: List of lists containing the associated variance values.

    Raises:
        HTTPException: If a gRPC communication error occurs, converted into a 500
                       Internal Server Error.
    """
    try:
        with grpc.insecure_channel(PREDICTOR_SERVICE_CHANNEL) as channel:
            debug(
                "Predictor service call started",
                extra={
                    "features_len": len(features),
                    "keys_seq_len": len(keys_seq),
                    "features_shape": features_shape,
                    "keys_shape": keys_shape,
                    "rollout_horizon": api_config.kwargs.rollout_horizon.value,
                    "mc_dropout_samples": api_config.kwargs.mc_dropout_samples.value,
                    "unbiased_variance": api_config.kwargs.unbiased_variance.value,
                    "time_step_increment": api_config.kwargs.time_step_increment.value,
                    "context": "Predictor service call",
                },
            )

            # Create stub, build request for the service,
            # and call it retrieving the response
            stub = pb2_grpc.PredictorServiceStub(channel)
            request = pb2.PredictorServiceRequest(
                features=features,
                keys_seq=keys_seq,
                features_shape=features_shape,
                keys_shape=keys_shape,
                rollout_horizon=api_config.kwargs.rollout_horizon.value,
                mc_dropout_samples=api_config.kwargs.mc_dropout_samples.value,
                unbiased_variance=api_config.kwargs.unbiased_variance.value,
                time_step_increment=api_config.kwargs.time_step_increment.value,
            )
            response = stub.Predict(request)

            # Prepare outputs to return
            outputs = [fl.values for fl in response.outputs]
            variances = [fl.values for fl in response.variances]

            debug(
                "Predictor service call completed",
                extra={
                    "outputs_num": len(outputs),
                    "variances_num": len(variances),
                    "context": "Predictor service call",
                },
            )

            return outputs, variances
    except grpc.RpcError as e:
        error(
            "Predictor service call failed",
            extra={
                "exception": str(e),
                "features_len": len(features),
                "keys_seq_len": len(keys_seq),
                "features_shape": features_shape,
                "keys_shape": keys_shape,
                "rollout_horizon": api_config.kwargs.rollout_horizon.value,
                "mc_dropout_samples": api_config.kwargs.mc_dropout_samples.value,
                "unbiased_variance": api_config.kwargs.unbiased_variance.value,
                "time_step_increment": api_config.kwargs.time_step_increment.value,
                "context": "Predictor service call",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
