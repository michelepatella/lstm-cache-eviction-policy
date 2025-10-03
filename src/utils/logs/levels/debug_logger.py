import logging
from typing import Any

from utils.logs.levels.error_logger import error
from utils.logs.initializer import logs_phase


def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a debug-level message with contextual phase.

    Parameters:
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
        logging.debug(
            msg,
            *args,
            extra={"phase": logs_phase.get()},
            **kwargs,
        )
    except (TypeError, ValueError, KeyError) as e:
        msg_err = "Failed to log debug message"
        error("%s: %s", msg_err, e)
        raise RuntimeError(msg_err) from e
