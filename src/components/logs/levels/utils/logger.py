import logging
from typing import Any, Optional

from components.logs.initializer import logs_phase
from const import LOGS_PHASE_DEFAULT, LOGS_PHASE_NAME


def log(
    level: int,
    msg: str,
    *args: Any,
    log_phase_name: Optional[str] = LOGS_PHASE_NAME,
    log_phase: Optional[str] = LOGS_PHASE_DEFAULT,
    **kwargs: Any
):
    """
    Log a message.

    This function logs a level-provided contextual message, using log phase
    as context.

    Args:
        level (int): The log level.
        msg (str): The message to log.
        args (Any): Positional arguments for the message.
        log_phase_name (Optional[str]): The name of the log phase.
        log_phase (Optional[str]): Current log phase.
        kwargs (Any): Keyword arguments for the logging function.

    Returns:
        None
    """
    # Retrieve current log phase from
    # contextual variable if None is passed
    if log_phase is None:
        log_phase = logs_phase.get()

    # Log message using provided level
    logging.log(
        level,
        msg,
        *args,
        extra={log_phase_name: log_phase},
        **kwargs,
    )
