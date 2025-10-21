import logging
from typing import Any

from components.const import LOGS_DEFAULT_PHASE, LOGS_PHASE_NAME
from components.logs.levels.utils.logger import log


def info(
    msg: str,
    *args: Any,
    log_phase_name: str = LOGS_PHASE_NAME,
    log_phase: str = LOGS_DEFAULT_PHASE,
    **kwargs: Any
) -> None:
    """
    Log an info-level message.

    This function logs a contextual info-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        args (Any): Positional arguments for the message.
        log_phase_name (str): The name of the log phase.
        log_phase (str): Current log phase.
        kwargs (Any): Keyword arguments for the logging function.

    Returns:
        None
    """
    log(logging.INFO, msg, *args, log_phase_name, log_phase, **kwargs)
