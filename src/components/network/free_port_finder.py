"""free_port_finder.py

Module providing utility functions to dynamically locate and reserve a free
TCP port on the local machine.

This module is used for configuring services, especially those involved
in distributed computing (like PyTorch DDP), where unique and available
network endpoints are necessary for inter-process communication.

Functions:
    find_free_port() -> int
        Locates and returns an unused TCP port number.
"""

import socket

from components.const import (
    NETWORK_FIND_FREE_PORT_AUTO,
    NETWORK_FIND_FREE_PORT_INTERFACES,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def find_free_port() -> int:
    """Finds an unused, available TCP port number on the local machine.

    This function works by creating a temporary TCP socket, binding the
    socket to a specified interface, and requesting the OS to dynamically
    assign a free port. After having retrieved the free port number,
    the socket is closed to free the port, which is finally returned.

    Returns:
        int: An integer representing the free TCP port number.

    Raises:
        RuntimeError: If free port finding fails:
            * System-level network errors, permission issues, or lack of
              available ports (OSError).
    """
    try:
        debug(
            "Free port finding started",
            extra={"context": "Free port finding"},
        )

        # Create a new TCP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the specified
        # interface and port
        s.bind(
            (NETWORK_FIND_FREE_PORT_INTERFACES, NETWORK_FIND_FREE_PORT_AUTO),
        )

        # Retrieve the actual port assigned
        # by the operating system
        _, port = s.getsockname()

        # Close the socket to free
        # up the port
        s.close()

        debug(
            "Free port finding completed",
            extra={"port": port, "context": "Free port finding"},
        )

        return port
    except OSError as e:
        msg = "Free port finding failed"
        error(msg, extra={"exception": str(e), "context": "Free port finding"})
        raise RuntimeError(msg) from e
