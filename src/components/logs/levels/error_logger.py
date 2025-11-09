"""error_logger.py

Error-level logging utility.

This module provides a convenience function `error` for logging
messages at the error level. It integrates with the application's
structured logging system, allowing extra fields.

Functions:
    error(
        msg: str,
        extra: dict[str, Any] | None = None,
    ) -> None
        Logs a message at error level with optional context.
"""

import logging
from typing import Any

from components.logs.levels.utils.logger import log


def error(
    msg: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log an error-level message.

    This function logs a contextual error-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        extra (dict[str, Any] | None): Optional additional context.

    Returns:
        None
    """
    log(logging.ERROR, msg, extra)
