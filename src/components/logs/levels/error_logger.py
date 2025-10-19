import logging
from typing import Any, Optional

from components.logs.levels.utils.logger import log
from const import LOGS_PHASE_DEFAULT, LOGS_PHASE_NAME


def error(
    msg: str,
    *args: Any,
    log_phase_name: Optional[str] = LOGS_PHASE_NAME,
    log_phase: Optional[str] = LOGS_PHASE_DEFAULT,
    **kwargs: Any
) -> None:
    """
    Log an error-level message.

    This function logs a contextual error-level message, using log phase
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
    log(logging.ERROR, msg, *args, log_phase_name, log_phase, **kwargs)
