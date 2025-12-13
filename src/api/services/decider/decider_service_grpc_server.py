"""decider_service_grpc_server.py

Module responsible for starting the gRPC server that hosts the
Decider Service implementation.

This service handles eviction decisions based on key scores and cache state.
It configures a ThreadPoolExecutor to handle concurrent requests and binds
the service implementation to a specified channel, keeping the server running
indefinitely.

Functions:
    decider_service_serve() -> None: Initializes and starts the gRPC server.
"""

from concurrent import futures

import grpc

import api.services.decider.decider_service_pb2_grpc as pb2_grpc
from api.const import DECIDER_SERVICE_CHANNEL
from api.services.decider.decider_service import DeciderService


def decider_service_serve() -> None:
    """Initializes and starts the gRPC server for the Decider Service.

    The server is configured using a thread pool executor for efficient
    concurrency. It registers the concrete implementation of the
    Decider Service and binds itself to the predefined channel.
    The function then blocks the main thread, waiting for the server to
    receive and process requests until termination is signaled.
    """
    server = grpc.server(futures.ThreadPoolExecutor())
    pb2_grpc.add_DeciderServiceServicer_to_server(
        DeciderService(),
        server,
    )
    server.add_insecure_port(DECIDER_SERVICE_CHANNEL)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    decider_service_serve()
