"""debug_logger.py

Debug-level logging utility.

This module provides a convenience function `debug` for logging
messages at the debug level. It integrates with the application's
structured logging system, allowing optional extra fields.

Functions:
    debug(
        msg: str,
        extra: dict[str, Any] | None = None,
    ) -> None
        Logs a message at debug level with optional context.
"""

import logging
from typing import Any

from components.logs.levels.utils.logger import log


def debug(
    msg: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a debug-level message.

    This function logs a contextual debug-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        extra (dict[str, Any] | None): Optional additional context.

    Returns:
        None
    """
    log(logging.DEBUG, msg, extra)
