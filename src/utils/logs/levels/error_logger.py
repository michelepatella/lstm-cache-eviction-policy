import logging
from typing import Any

from utils.logs.initializer import logs_phase


def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log an error-level message with the current phase context.

    Parameters:
        msg (str): The message to log.
        args (Any): Positional arguments for message formatting.
        kwargs (Any): Keyword arguments for the logging function.

    Returns:
        None

    Raises:
        RuntimeError: If logging fails, e.g. due to:
            * Invalid message formatting.
            * Problems applying extra context.
    """
    try:
        logging.error(
            msg,
            *args,
            extra={"phase": logs_phase.get()},
            **kwargs,
        )
    except (TypeError, ValueError, KeyError) as e:
        msg_err = "Failed to log error message"
        raise RuntimeError(msg_err) from e
