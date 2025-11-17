"""logger.py

Core logging function for structured logging.

This module provides the `log` function, which is the underlying
utility used by all logging level helpers (debug, info, error, etc.).
It standardizes the way messages are logged, attaching the extra context
to the log record.

Functions:
    log(
        level: int,
        msg: str,
        extra: dict[str, Any] | None = None,
    ) -> None
        Logs a message at the specified level with optional context.
"""

import logging
from typing import Any

from src.const import LOGS_LOGGER_NAME


def log(
    level: int,
    msg: str,
    extra: dict[str, Any] | None = None,
):
    """Log a message.

    This function logs a level-provided contextual message, using log phase
    as context.

    Args:
        level (int): The log level.
        msg (str): The message to log.
        extra (dict[str, Any] | None): Optional additional context.

    Returns:
        None
    """
    # Log message using provided level
    logger = logging.getLogger(LOGS_LOGGER_NAME)
    logger.log(
        level,
        msg,
        extra=extra,
    )
