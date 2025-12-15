"""scorer_service_grpc_server.py

Module responsible for starting the gRPC server that hosts the
Scorer Service implementation.

It configures a ThreadPoolExecutor to handle incoming requests and binds
the service implementation to a specified channel, keeping the server
running indefinitely.

Functions:
    scorer_service_serve() -> None: Initializes and starts the gRPC server.
"""

from concurrent import futures

import grpc

import api.services.scorer.scorer_service_pb2_grpc as pb2_grpc
from api.const import SCORER_SERVICE_CHANNEL
from api.services.scorer.scorer_service import ScorerService


def scorer_service_serve() -> None:
    """Initializes and starts the gRPC server for the Scorer Service.

    The server is configured to use a thread pool executor for concurrency.
    It registers the concrete implementation of the Scorer Service and binds
    itself to the predefined channel. The function then blocks, waiting for
    the server to terminate.

    Returns:
        None
    """
    server = grpc.server(futures.ThreadPoolExecutor())
    pb2_grpc.add_ScorerServiceServicer_to_server(ScorerService(), server)
    server.add_insecure_port(SCORER_SERVICE_CHANNEL)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    scorer_service_serve()
