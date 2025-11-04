"""logger.py

Core logging function for structured logging.

This module provides the `log` function, which is the underlying
utility used by all logging level helpers (debug, info, error, etc.).
It standardizes the way messages are logged, attaching the current
log phase and any optional extra context to the log record.

Functions:
    log(
        level: int,
        msg: str,
        log_phase: str = LOGS_FIELD_PHASE_DEFAULT,
        extra: dict[str, Any] | None = None,
        logger_name: str = LOGS_LOGGER_NAME
    ) -> None
        Logs a message at the specified level with optional context.
"""

import logging
from typing import Any

from components.const import (
    LOGS_FIELD_PHASE_DEFAULT,
    LOGS_FIELD_PHASE_NAME,
    LOGS_LOGGER_NAME,
)
from components.logs.initializer import logs_phase


def log(
    level: int,
    msg: str,
    log_phase: str = LOGS_FIELD_PHASE_DEFAULT,
    extra: dict[str, Any] | None = None,
    logger_name: str = LOGS_LOGGER_NAME,
):
    """Log a message.

    This function logs a level-provided contextual message, using log phase
    as context.

    Args:
        level (int): The log level.
        msg (str): The message to log.
        log_phase (str): Current log phase.
        extra (dict[str, Any] | None): Optional additional context.
        logger_name (str): The name of the logger to use.

    Returns:
        None
    """
    # Retrieve current log phase
    if not log_phase or log_phase == LOGS_FIELD_PHASE_DEFAULT:
        log_phase = logs_phase.get()

    # Prepare extra section as dictionary
    extra_dict = {LOGS_FIELD_PHASE_NAME: log_phase}
    extra_dict.update(extra)

    # Log message using provided level
    logger = logging.getLogger(logger_name)
    logger.log(
        level,
        msg,
        extra=extra_dict,
    )
