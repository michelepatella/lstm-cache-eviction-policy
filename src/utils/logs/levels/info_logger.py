import logging
from typing import Any

from pipeline.const import LOGS_PHASE_NAME
from utils.logs.initializer import logs_phase


def info(
    msg: str,
    *args: Any,
    log_phase_name: str = LOGS_PHASE_NAME,
    log_phase: str = None,
    **kwargs: Any
) -> None:
    """
    Log an info-level message with the current phase context.

    Args:
        msg (str): The message to log.
        args (Any): Positional arguments for message formatting.
        log_phase_name (str): The name of the log phase.
        log_phase (str): Current log phase.
        kwargs (Any): Keyword arguments for the logging function.

    Returns:
        None
    """
    # Retrieve current log phase
    # if None is passed
    if log_phase is None:
        log_phase = logs_phase.get()

    logging.info(
        msg,
        *args,
        extra={log_phase_name: log_phase},
        **kwargs,
    )
