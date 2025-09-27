import contextvars
import logging
from logging.handlers import RotatingFileHandler

from const import (
    LOGS_BACKUP_COUNT,
    LOGS_DEFAULT_LEVEL,
    LOGS_FORMAT,
    LOGS_MAX_BYTES,
    LOGS_SAVE_PATH,
)

# Contextual variable indicating
# the current phase
logs_phase = contextvars.ContextVar("phase", default="unknown")


def setup_logs() -> None:
    """
    Setup global logging configuration.

    This function configures logging to:
        - Write all INFO-level messages to a rotating file.
        - Show INFO-level messages on the terminal.
        - Use a consistent log format.

    Returns:
        None
    """
    # Create formatter for log messages
    formatter = logging.Formatter(LOGS_FORMAT)

    # File handler for logging to
    # file with rotation
    file_handler = RotatingFileHandler(
        LOGS_SAVE_PATH,
        maxBytes=LOGS_MAX_BYTES,
        backupCount=LOGS_BACKUP_COUNT,
    )
    file_handler.setLevel(LOGS_DEFAULT_LEVEL)
    file_handler.setFormatter(formatter)

    # Stream handler for terminal output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOGS_DEFAULT_LEVEL)
    stream_handler.setFormatter(formatter)

    # Configure the root logger
    logging.basicConfig(
        level=LOGS_DEFAULT_LEVEL,
        handlers=[file_handler, stream_handler],
    )
