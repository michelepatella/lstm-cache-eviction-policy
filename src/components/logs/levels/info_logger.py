import logging
from typing import Any

from components.const import LOGS_FIELD_PHASE_DEFAULT, LOGS_LOGGER_NAME
from components.logs.levels.utils.logger import log


def info(
    msg: str,
    log_phase: str = LOGS_FIELD_PHASE_DEFAULT,
    extra: dict[str, Any] | None = None,
    logger_name: str = LOGS_LOGGER_NAME,
) -> None:
    """Log an info-level message.

    This function logs a contextual info-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        log_phase (str): Current log phase.
        extra (dict[str, Any] | None): Optional additional context.
        logger_name (str): The name of the logger to use.

    Returns:
        None
    """
    log(logging.INFO, msg, log_phase, extra, logger_name)
