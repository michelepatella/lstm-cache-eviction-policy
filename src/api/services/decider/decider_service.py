"""decider_service.py

This module implements the gRPC servicer for the Decider Service.

The Decider Service is responsible for making the final decision on cache
eviction. It processes current cache state, calculated scores, and
exclusions to return the specific keys that should be evicted.

Classes:
    DeciderService: gRPC Servicer class implementing the Decide method.
"""

import grpc

import api.services.decider.decider_service_pb2 as pb2
import api.services.decider.decider_service_pb2_grpc as pb2_grpc
from components.caches.implementations.items.evictions.score_based_evictor import (
    evict_score_based_items,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


class DeciderService(pb2_grpc.DeciderServiceServicer):
    """gRPC Servicer that implements the Decider Service definition.

    This class serves as the interface between the Gateway API and
    the eviction logic, receiving scored keys and constraints via gRPC
    and returning the list of keys to be evicted.
    """

    def Decide(
        self: "DeciderService",
        request: pb2.DeciderServiceRequest,
        context: grpc.ServicerContext,
    ) -> pb2.DeciderServiceResponse:
        """Determines keys to evict based on provided scores and constraints.

        This method processes the request containing keys currently in cache,
        their associated scores, keys to exclude from eviction, and the
        target number of evictions. It delegates the logic to the
        score-based eviction logic.

        Args:
            self (DeciderService): The instance of the servicer class.
            request (pb2.DeciderServiceRequest): gRPC request containing
                                                 keys in cache, scores,
                                                 excluded keys, and
                                                 number of evictions.
            context (grpc.ServicerContext): gRPC context object for setting
                                            status codes and details in case
                                            of failure.

        Returns:
            pb2.DeciderServiceResponse: A gRPC response containing the list
                                        of keys selected for eviction.
        """
        try:
            info(
                "Decider service started",
                extra={
                    "keys_in_cache_num": len(request.keys_in_cache),
                    "key_scores_num": len(request.key_scores),
                    "excluded_keys_num": len(request.excluded_keys),
                    "evictions_num": request.num_evictions,
                    "context": "Decider service",
                },
            )

            # Decide which key(s) to evict based on
            # scores, as well as keys currently in cache,
            # keys to exclude from eviction, and number
            # of key to evict
            keys_to_evict = evict_score_based_items(
                list(request.keys_in_cache),
                list(request.key_scores),
                list(request.excluded_keys),
                request.num_evictions,
            )

            info(
                "Decider service completed",
                extra={
                    "keys_to_evict_num": len(keys_to_evict),
                    "context": "Decider service",
                },
            )

            # Response
            return pb2.DeciderServiceResponse(keys_to_evict=keys_to_evict)

        except Exception as e:
            error(
                "Decider service failed",
                extra={
                    "keys_in_cache_num": len(request.keys_in_cache),
                    "key_scores_num": len(request.key_scores),
                    "excluded_keys_num": len(request.excluded_keys),
                    "evictions_num": request.num_evictions,
                    "context": "Decider service",
                },
            )
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return pb2.FeaturizerServiceResponse()
