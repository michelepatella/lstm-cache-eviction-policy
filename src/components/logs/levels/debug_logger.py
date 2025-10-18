import logging
from typing import Any, Optional

from components.logs.levels.utils.logger import log
from const import LOGS_PHASE_NAME, LOGS_PHASE_DEFAULT


def debug(
    msg: str,
    *args: Any,
    log_phase_name: Optional[str] = LOGS_PHASE_NAME,
    log_phase: Optional[str] = LOGS_PHASE_DEFAULT,
    **kwargs: Any
) -> None:
    """
    Log a debug-level message.

    This function logs a contextual debug-level message, using log phase
    as context.

    Args:
        msg (str): The message to log.
        args (Any): Positional arguments for the message.
        log_phase_name (Optional[str]): The name of the log phase.
        log_phase (Optional[str]): Current log phase.
        kwargs (Any): Keyword arguments for the logging function.

    Returns:
        None
    """
    log(logging.DEBUG, msg, *args, log_phase_name, log_phase, **kwargs)
