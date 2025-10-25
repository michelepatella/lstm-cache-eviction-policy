import logging
from typing import Optional, Dict, Any

from components.const import (
    LOGS_DEFAULT_PHASE,
    LOGS_PHASE_NAME,
    LOGS_LOGGER_NAME,
)
from components.logs.initializer import logs_phase


def log(
    level: int,
    msg: str,
    log_phase_name: str = LOGS_PHASE_NAME,
    log_phase: str = LOGS_DEFAULT_PHASE,
    extra: Optional[Dict[str, Any]] = None,
    logger_name: str = LOGS_LOGGER_NAME,
):
    """
    Log a message.

    This function logs a level-provided contextual message, using log phase
    as context.

    Args:
        level (int): The log level.
        msg (str): The message to log.
        log_phase_name (str): The name of the log phase.
        log_phase (str): Current log phase.
        extra (Optional[Dict[str, Any]]): Optional additional context.
        logger_name (str): The name of the logger to use.

    Returns:
        None
    """
    # Retrieve current log phase from
    # contextual variable if None is passed
    if log_phase is None:
        log_phase = logs_phase.get()

    # Prepare extra section as dictionary
    extra_dict = {log_phase_name: log_phase}
    extra_dict.update(extra)

    # Log message using provided level
    logger = logging.getLogger(logger_name)
    logger.log(
        level,
        msg,
        extra=extra_dict,
    )
