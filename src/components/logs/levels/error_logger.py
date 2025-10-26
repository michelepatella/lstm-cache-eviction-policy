import logging
from typing import Any

from components.const import LOGS_DEFAULT_PHASE, LOGS_PHASE_NAME
from components.logs.levels.utils.logger import log


def error(
    msg: str,
    log_phase_name: str = LOGS_PHASE_NAME,
    log_phase: str = LOGS_DEFAULT_PHASE,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log an error-level message.

    This function logs a contextual error-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        log_phase_name (str): The name of the log phase.
        log_phase (str): Current log phase.
        extra (dict[str, Any] | None): Optional additional context.

    Returns:
        None
    """
    log(logging.ERROR, msg, log_phase_name, log_phase, extra)
