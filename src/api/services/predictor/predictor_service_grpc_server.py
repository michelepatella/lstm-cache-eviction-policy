"""predictor_service_grpc_server.py

Module responsible for starting the gRPC server that hosts the
Predictor Service implementation.

It configures a ThreadPoolExecutor to handle incoming requests and binds
the service implementation to a specified channel, keeping the server
running indefinitely.

Functions:
    predictor_service_serve() -> None: Initializes and starts the gRPC server.
"""

from concurrent import futures

import grpc
import predictor_service_pb2_grpc as pb2_grpc

from api.const import PREDICTOR_SERVICE_CHANNEL
from api.services.predictor.predictor_service import PredictorService


def predictor_service_serve() -> None:
    """Initializes and starts the gRPC server for the Predictor Service.

    The server is configured to use a thread pool executor for concurrency.
    It registers the concrete implementation of the Predictor Service and
    binds itself to the predefined channel. The function then blocks,
    waiting for the server to terminate.

    Returns:
        None
    """
    server = grpc.server(futures.ThreadPoolExecutor())
    pb2_grpc.add_PredictorServiceServicer_to_server(PredictorService(), server)
    server.add_insecure_port(PREDICTOR_SERVICE_CHANNEL)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    predictor_service_serve()
