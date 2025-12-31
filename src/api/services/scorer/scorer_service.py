"""scorer_service.py

This module implements the gRPC servicer for the Scorer Service.

The Scorer Service is responsible for calculating importance scores for
data items. It processes model outputs and variances using confidence-based
logic to return a final score for each key, which dictates eviction priority.

Classes:
    ScorerService: gRPC Servicer class implementing the Score method.
"""

import grpc
import numpy as np

import api.services.scorer.scorer_service_pb2 as pb2
import api.services.scorer.scorer_service_pb2_grpc as pb2_grpc
from components.caches.implementations.items.scores.survival_uncertainty_calculator import (
    calculate_survival_uncertainty_scores,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


class ScorerService(pb2_grpc.ScorerServiceServicer):
    """gRPC Servicer that implements the Scorer Service definition.

    This class serves as the interface between the API Gateway and the
    scoring logic, receiving prediction data via gRPC and returning
    calculated item scores.
    """

    def Score(
        self: "ScorerService",
        request: pb2.ScorerServiceRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ScorerServiceResponse:
        """Computes item scores based on prediction outputs and confidence.

        This method extracts model outputs and variances from the gRPC request,
        along with configuration weights, and invokes the calculation logic to
        determine a final score for each key.

        Args:
            self (ScorerService): The instance of the servicer class.
            request (pb2.ScorerServiceRequest): gRPC request containing model
                                                outputs, variances, and
                                                weighting parameters.
            context (grpc.ServicerContext): gRPC context object for setting
                                            status codes and details in case
                                            of failure.

        Returns:
            pb2.ScorerServiceResponse: A gRPC response containing the list
                                       of calculated scores for each key.
        """
        try:
            debug(
                "Scorer service started",
                extra={
                    "outputs_num": len(request.outputs),
                    "variances_num": len(request.variances),
                    "conf_level": request.conf_level,
                    "conf_weight": request.conf_weight,
                    "context": "Scorer service",
                },
            )

            # Extract outputs and variances as
            # a list of np.array
            outputs = [np.array(request.outputs)]
            variances = [np.array(request.variances)]

            # Calculate key scores
            key_scores = calculate_survival_uncertainty_scores(
                outputs,
                variances,
                request.conf_level,
                request.conf_weight,
            )

            debug(
                "Scorer service completed",
                extra={
                    "key_scores": key_scores.tolist(),
                    "context": "Scorer service",
                },
            )

            # Response
            return pb2.ScorerServiceResponse(key_scores=key_scores.tolist())

        except Exception as e:
            error(
                "Scorer service failed",
                extra={
                    "exception": str(e),
                    "outputs_num": len(request.outputs),
                    "variances_num": len(request.variances),
                    "conf_level": request.conf_level,
                    "conf_weight": request.conf_weight,
                    "context": "Scorer service",
                },
            )
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return pb2.ScorerServiceResponse()
