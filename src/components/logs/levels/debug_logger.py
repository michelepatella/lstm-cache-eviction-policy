import logging
from typing import Any, Optional, Dict

from components.const import LOGS_DEFAULT_PHASE, LOGS_PHASE_NAME
from components.logs.levels.utils.logger import log


def debug(
    msg: str,
    log_phase_name: str = LOGS_PHASE_NAME,
    log_phase: str = LOGS_DEFAULT_PHASE,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a debug-level message.

    This function logs a contextual debug-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        log_phase_name (str): The name of the log phase.
        log_phase (str): Current log phase.
        extra (Optional[Dict[str, Any]]): Optional additional context.

    Returns:
        None
    """
    log(logging.DEBUG, msg, log_phase_name, log_phase, extra)
