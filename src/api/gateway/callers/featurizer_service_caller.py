"""featurizer_service_caller.py

Module containing the logic to call the gRPC Featurizer Service.

This module is responsible for communicating with the Featurizer Service.
It handles the serialization of request data (access sequences) into the
gRPC format, manages the gRPC channel connection, executes the remote
procedure call, and processes the response.

Functions:
    call_featurizer_service(
        last_accesses: list[tuple[float, int]],
    ) -> tuple[list[float], list[int], list[int], list[int]]:
        Initiates the gRPC call to the Featurizer Service to build features.
"""

import grpc
from fastapi import HTTPException, status

import api.services.featurizer.featurizer_service_pb2 as pb2
import api.services.featurizer.featurizer_service_pb2_grpc as pb2_grpc
from api.const import FEATURIZER_SERVICE_CHANNEL
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def call_featurizer_service(
    last_accesses: list[tuple[float, int]],
) -> tuple[list[float], list[int], list[int], list[int]]:
    """Initiates the gRPC call to the Featurizer Service to build features.

    This function opens a channel to the Featurizer Service, serializes the
    last accesses list into the appropriate gRPC request format, and calls
    the remote gRPC method.

    Args:
        last_accesses (list[tuple[float, int]]): A sequence of (timestamp, key)
                                                 tuples representing recent data
                                                 accesses to be converted into
                                                 features.

    Returns:
        tuple[list[float], list[int], list[int], list[int]]:
            - features: Flattened list of feature tensor values.
            - keys_seq: Flattened list of keys in sequence.
            - features_shape: List representing the original shape of the
                              feature tensor.
            - keys_shape: List representing the original shape of the
                          keys sequence.

    Raises:
        HTTPException: If a gRPC communication error occurs, converted into a 500
                       Internal Server Error.
    """
    try:
        with grpc.insecure_channel(FEATURIZER_SERVICE_CHANNEL) as channel:
            debug(
                "Featurizer service call started",
                extra={
                    "last_accesses_num": len(last_accesses),
                    "context": "Featurizer service call",
                },
            )

            # Create stub, build request for the service,
            # and call it retrieving the response
            stub = pb2_grpc.FeaturizerServiceStub(channel)
            request = pb2.FeaturizerServiceRequest(
                last_accesses=[
                    pb2.LastAccess(timestamp=ts, key=k)
                    for ts, k in last_accesses
                ],
            )
            response = stub.Build(request)

            debug(
                "Featurizer service call completed",
                extra={
                    "features_num": len(response.features),
                    "keys_seq_num": len(response.keys_seq),
                    "features_shape": response.features_shape,
                    "keys_shape": response.keys_shape,
                    "context": "Featurizer service call",
                },
            )

            return (
                response.features,
                response.keys_seq,
                response.features_shape,
                response.keys_shape,
            )

    except grpc.RpcError as e:
        error(
            "Featurizer service call failed",
            extra={
                "exception": str(e),
                "last_accesses_num": len(last_accesses),
                "context": "Featurizer service call",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
