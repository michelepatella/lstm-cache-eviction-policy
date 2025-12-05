"""info_logger.py

Info-level logging utility.

This module provides a convenience function `info` for logging
messages at the info level. It integrates with the application's
structured logging system, allowing extra fields.

Functions:
    info(
        msg: str,
        extra: dict[str, Any] | None = None,
    ) -> None
        Logs a message at info level with optional context.
"""

import logging
from typing import Any

from components.logs.levels.utils.logger import log


def info(
    msg: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log an info-level message.

    This function logs a contextual info-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        extra (dict[str, Any] | None): Optional additional context.

    Returns:
        None
    """
    log(logging.INFO, msg, extra)
