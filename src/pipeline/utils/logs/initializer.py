import contextvars
import logging
from logging.handlers import RotatingFileHandler

from const import (
    LOGS_CONSOLE_LEVEL,
    LOGS_DEBUG_FILE_PATH,
    LOGS_ERROR_FILE_PATH,
    LOGS_FILE_BACKUP_COUNT,
    LOGS_FILE_BASE_LEVEL,
    LOGS_FILE_MAX_BYTES,
    LOGS_FORMAT,
    LOGS_INFO_FILE_PATH,
    LOGS_PHASE_DEFAULT,
    LOGS_PHASE_NAME,
)

# Contextual variable indicating
# the current phase
logs_phase = contextvars.ContextVar(
    LOGS_PHASE_NAME, default=LOGS_PHASE_DEFAULT
)


def initialize_logs() -> None:
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

    # File handlers for logging to file
    debug_file_handler = RotatingFileHandler(
        LOGS_DEBUG_FILE_PATH,
        maxBytes=LOGS_FILE_MAX_BYTES,
        backupCount=LOGS_FILE_BACKUP_COUNT,
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(formatter)

    info_file_handler = RotatingFileHandler(
        LOGS_INFO_FILE_PATH,
        maxBytes=LOGS_FILE_MAX_BYTES,
        backupCount=LOGS_FILE_BACKUP_COUNT,
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)

    error_file_handler = RotatingFileHandler(
        LOGS_ERROR_FILE_PATH,
        maxBytes=LOGS_FILE_MAX_BYTES,
        backupCount=LOGS_FILE_BACKUP_COUNT,
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    # Stream handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOGS_CONSOLE_LEVEL)
    console_handler.setFormatter(formatter)

    # Configure the root logger
    logging.basicConfig(
        level=LOGS_FILE_BASE_LEVEL,
        handlers=[
            console_handler,
            debug_file_handler,
            info_file_handler,
            error_file_handler,
        ],
    )
