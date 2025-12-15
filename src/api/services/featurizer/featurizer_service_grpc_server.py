"""featurizer_service_grpc_server.py

Module responsible for starting the gRPC server that hosts the
Featurizer Service implementation.

This service handles the transformation of raw data into structured features
suitable for model consumption. It configures a ThreadPoolExecutor to handle
concurrent requests and binds the service implementation to a specified channel,
keeping the server running indefinitely.

Functions:
    featurizer_service_serve() -> None: Initializes and starts the gRPC server.
"""

from concurrent import futures

import grpc

import api.services.featurizer.featurizer_service_pb2_grpc as pb2_grpc
from api.const import FEATURIZER_SERVICE_CHANNEL
from api.services.featurizer.featurizer_service import FeaturizerService


def featurizer_service_serve() -> None:
    """Initializes and starts the gRPC server for the Featurizer Service.

    The server is configured using a thread pool executor for efficient
    concurrency. It registers the concrete implementation of the
    Featurizer Service and binds itself to the predefined channel.
    The function then blocks the main thread, waiting for the server to
    receive and process requests until termination is signaled.
    """
    server = grpc.server(futures.ThreadPoolExecutor())
    pb2_grpc.add_FeaturizerServiceServicer_to_server(
        FeaturizerService(),
        server,
    )
    server.add_insecure_port(FEATURIZER_SERVICE_CHANNEL)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    featurizer_service_serve()
