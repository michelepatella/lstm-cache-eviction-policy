"""predictor_service.py

This module implements the gRPC servicer for the Predictor Service.

The Predictor Service is responsible for executing the loaded PyTorch model
to generate outputs and associated uncertainty estimates (variances) for the
cache key access probabilities.

Classes:
    PredictorService: gRPC Servicer class implementing the Predict method.
"""

import grpc
import mlflow
import torch

import api.services.predictor.predictor_service_pb2 as pb2
import api.services.predictor.predictor_service_pb2_grpc as pb2_grpc
from api.config.pydantic.api_config import APIConfig
from api.const import (
    API_CONFIG_FILE_PATH,
    MLFLOW_TRACKING_URI,
)
from components.const import TORCH_DTYPE_FEATURES, TORCH_DTYPE_TARGET
from components.device.mover import move_to_device
from components.device.selector import select_device
from components.inference.autoregressive_rollout.runner import (
    compute_autoregressive_rollout,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.yaml.io.loader import load_yaml
from const import (
    MLFLOW_MODEL_SIMULATION_NAME,
)

# ----------------------------
# Setup
# ----------------------------
# Load API configuration
api_config_file = load_yaml(API_CONFIG_FILE_PATH)
api_config = APIConfig(**api_config_file)

# Prepare production model environment:
# select device and set configured quantization engine
device = select_device(api_config.resources.device.type)
torch.backends.quantized.engine = (
    api_config.model.optimizations.quantization.engine
)

# Load the last version of the production model
mlflow_client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
model_versions = mlflow_client.search_model_versions(
    f"name='{MLFLOW_MODEL_SIMULATION_NAME}'",
)
last_model_version = max(
    (v for v in model_versions),
    key=lambda v: int(v.version),
    default=None,
)
if last_model_version is not None:
    model = mlflow.pytorch.load_model(
        model_uri=f"models:/{MLFLOW_MODEL_SIMULATION_NAME}/{last_model_version.version}",
    )


class PredictorService(pb2_grpc.PredictorServiceServicer):
    """gRPC Servicer that implements the Predictor Service definition.

    This class provides the inference functionality, running an autoregressive
    rollout based on the input features.
    """

    def Predict(
        self: "PredictorService",
        request: pb2.PredictorServiceRequest,
        context: grpc.ServicerContext,
    ) -> pb2.PredictorServiceResponse:
        """Generates future access probability predictions and variances.

        This method receives features, keys, and hyperparameters via gRPC,
        converts them to tensors, moves them to the appropriate device,
        and runs the autoregressive prediction pipeline.

        Args:
            self (PredictorService): The instance of the servicer class.
            request (pb2.PredictorServiceRequest): gRPC request containing
                                                   features, keys sequence,
                                                   and inference hyperparameters.
            context (grpc.ServicerContext): gRPC context object for setting
                                            status codes and details in case
                                            of failure.

        Returns:
            pb2.PredictorServiceResponse: A gRPC response containing the list of
                                          predicted outputs and their associated
                                          variances.
        """
        try:
            info(
                "Predictor service started",
                extra={
                    "rollout_horizon": request.rollout_horizon,
                    "mc_dropout_samples": request.mc_dropout_samples,
                    "time_step_increment": request.time_step_increment,
                    "context": "Predictor service",
                },
            )

            # Convert features and keys to tensors with
            # correct shape and move to device
            features_seq = torch.tensor(
                request.features,
                dtype=TORCH_DTYPE_FEATURES,
            ).reshape(tuple(request.features_shape))
            keys_seq = torch.tensor(
                request.keys_seq,
                dtype=TORCH_DTYPE_TARGET,
            ).reshape(tuple(request.keys_shape))
            features_seq = move_to_device(features_seq, device)
            keys_seq = move_to_device(keys_seq, device)

            # Compute autoregressive rollout
            all_outputs, all_variances = compute_autoregressive_rollout(
                model,
                features_seq,
                keys_seq,
                device,
                request.rollout_horizon,
                request.mc_dropout_samples,
                request.unbiased_variance,
                request.time_step_increment,
            )

            info(
                "Predictor service completed",
                extra={
                    "outputs_num": len(all_outputs),
                    "variances_num": len(all_variances),
                    "context": "Predictor service",
                },
            )

            # Response
            return pb2.PredictorServiceResponse(
                outputs=[
                    pb2.FloatList(values=o.tolist()) for o in all_outputs
                ],
                variances=[
                    pb2.FloatList(values=v.tolist()) for v in all_variances
                ],
            )

        except Exception as e:
            error(
                "Predictor service failed",
                extra={
                    "exception": str(e),
                    "rollout_horizon": request.rollout_horizon,
                    "mc_dropout_samples": request.mc_dropout_samples,
                    "time_step_increment": request.time_step_increment,
                    "context": "Predictor service",
                },
            )
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return pb2.PredictorServiceResponse()
