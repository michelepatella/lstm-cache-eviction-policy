"""decider_service_caller.py

Module containing the logic to call the gRPC Decider Service.

This module is responsible for communicating with the Decider Service.
It handles the serialization of cache state data (keys, scores, exclusions)
into the gRPC format, manages the gRPC channel connection, executes the
remote procedure call, and processes the response.

Functions:
    call_decider_service(
        keys_in_cache: list[int],
        key_scores: list[float],
        excluded_keys: list[int],
        num_evictions: int,
    ) -> list[int]:
        Initiates the gRPC call to the Decider Service to select keys for eviction.
"""

import grpc
from fastapi import HTTPException, status

import api.services.decider.decider_service_pb2 as pb2
import api.services.decider.decider_service_pb2_grpc as pb2_grpc
from api.const import DECIDER_SERVICE_CHANNEL
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def call_decider_service(
    keys_in_cache: list[int],
    key_scores: list[float],
    excluded_keys: list[int],
    num_evictions: int,
) -> list[int]:
    """Initiates the gRPC call to the Decider Service to select keys
    for eviction.

    This function opens a channel to the Decider Service, serializes the
    current cache state and constraints into the appropriate gRPC request
    format, and calls the remote gRPC method to determine evictions.

    Args:
        keys_in_cache (list[int]): List of keys currently present in the cache.
        key_scores (list[float]): List of key scores.
        excluded_keys (list[int]): List of keys that must be preserved and
                                   cannot be selected for eviction.
        num_evictions (int): The number of keys to evict.

    Returns:
        list[int]: A list of keys selected for eviction.

    Raises:
        HTTPException: If a gRPC communication error occurs, converted into
                       a 500 Internal Server Error.
    """
    try:
        with grpc.insecure_channel(DECIDER_SERVICE_CHANNEL) as channel:
            debug(
                "Decider service call started",
                extra={
                    "keys_in_cache_num": len(keys_in_cache),
                    "key_scores_num": len(key_scores),
                    "excluded_keys_num": len(excluded_keys),
                    "evictions_num": num_evictions,
                    "context": "Decider service call",
                },
            )

            # Create stub, build request for the service,
            # and call it retrieving the response
            stub = pb2_grpc.DeciderServiceStub(channel)
            request = pb2.DeciderServiceRequest(
                keys_in_cache=keys_in_cache,
                key_scores=key_scores,
                excluded_keys=excluded_keys,
                num_evictions=num_evictions,
            )
            response = stub.Decide(request)

            debug(
                "Decider service call completed",
                extra={
                    "keys_to_evict_num": len(list(response.keys_to_evict)),
                    "context": "Decider service call",
                },
            )

            return list(response.keys_to_evict)

    except grpc.RpcError as e:
        error(
            "Decider service call failed",
            extra={
                "exception": str(e),
                "keys_in_cache_num": len(keys_in_cache),
                "key_scores_num": len(key_scores),
                "excluded_keys_num": len(excluded_keys),
                "evictions_num": num_evictions,
                "context": "Decider service call",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
