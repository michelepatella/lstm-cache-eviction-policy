import logging
from typing import Any

from pipeline.const import LOGS_PHASE_NAME
from utils.logs.initializer import logs_phase
from utils.logs.levels.error_logger import error


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log an info-level message with contextual phase.

    Args:
        msg (str): The message to log.
        args (Any): Positional arguments for the message.
        kwargs (Any): Keyword arguments for the logging function.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs during logging, e.g.:
            * If message formatting fails.
            * If extra context cannot be applied.
    """
    try:
        logging.info(
            msg,
            *args,
            extra={LOGS_PHASE_NAME: logs_phase.get()},
            **kwargs,
        )
    except (TypeError, ValueError, KeyError) as e:
        msg_err = "Failed to log info message"
        error("%s: %s", msg_err, e)
        raise RuntimeError(msg_err) from e
