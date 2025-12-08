"""featurizer_service.py

This module implements the gRPC servicer for the Featurizer Service.

The Featurizer Service is responsible for preprocessing raw time-series data
(last accesses) and transforming it into features and keys sequence that are
required for the downstream model.

Classes:
    FeaturizerService: gRPC Servicer class implementing the Build method.
"""

import featurizer_service_pb2 as pb2
import featurizer_service_pb2_grpc as pb2_grpc
import grpc
import numpy as np

from api.config.pydantic.api_config import APIConfig
from api.const import API_CONFIG_FILE_PATH
from components.const import LIST_FIRST_IDX
from components.dataset.features.seq_builder import build_feature_seq
from components.device.selector import select_device
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.yaml.io.loader import load_yaml

# ----------------------------
# Setup
# ----------------------------
api_config_file = load_yaml(API_CONFIG_FILE_PATH)
api_config = APIConfig(**api_config_file)
device = select_device(api_config.resources.device.type)


class FeaturizerService(pb2_grpc.FeaturizerServiceServicer):
    """gRPC Servicer that implements the Featurizer Service definition.

    This class serves as the interface between the API Gateway and the
    feature engineering logic, receiving raw data via gRPC and returning
    model-ready features.
    """

    def Build(
        self: "FeaturizerService",
        request: pb2.FeaturizerServiceRequest,
        context: grpc.ServicerContext,
    ) -> pb2.FeaturizerServiceResponse:
        """Transforms raw access data into features and keys sequence for model.

        This method extracts timestamps and keys from the gRPC request,
        calls the feature engineering function to create features and
        keys sequence, and returns the flattened data along with the
        original shape information.

        Args:
            self (FeaturizerService): The instance of the servicer class.
            request (pb2.FeaturizerServiceRequest): gRPC request containing a
                                                    list of last accesses
                                                    (timestamp, key) pairs.
            context (grpc.ServicerContext): gRPC context object for setting
                                            status codes and details in case
                                             of failure.

        Returns:
            pb2.FeaturizerServiceResponse: A gRPC response containing the
                                           flattened features tensor, flattened
                                           keys sequence, and their respective
                                           shape meta-information.
        """
        try:
            info(
                "Featurizer service started",
                extra={
                    "last_accesses_num": len(request.last_accesses),
                    "context": "Featurizer service",
                },
            )

            # Extract timestamps and corresponding accessed keys
            # from last accesses data
            timestamps = np.array(
                request.last_accesses[LIST_FIRST_IDX].timestamps,
            )
            keys = np.array(request.last_accesses[LIST_FIRST_IDX].keys)

            # Build model-ready features and keys sequence
            features_seq, keys_seq = build_feature_seq(
                timestamps,
                keys,
                device,
            )

            info(
                "Featurizer service completed",
                extra={
                    "timestamps_num": len(timestamps),
                    "keys_num": len(keys),
                    "features_shape": list(features_seq.shape),
                    "keys_shape": list(keys_seq.shape),
                    "device": str(device),
                    "context": "Featurizer service",
                },
            )

            # Response
            return pb2.FeaturizerServiceResponse(
                features=features_seq.flatten().tolist(),
                keys_seq=keys_seq.flatten().tolist(),
                features_shape=list(features_seq.shape),
                keys_shape=list(keys_seq.shape),
            )

        except Exception as e:
            error(
                "Featurizer service failed",
                extra={
                    "exception": str(e),
                    "last_accesses_num": len(request.last_accesses),
                    "context": "Featurizer service",
                },
            )
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return pb2.FeaturizerServiceResponse()
