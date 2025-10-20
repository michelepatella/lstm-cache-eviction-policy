import contextvars
import logging
from typing import Optional

from components.logs.file_handlers.builder import build_logs_file_handler
from const import (
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

# Contextual variable for logging messages
logs_phase = contextvars.ContextVar(
    LOGS_PHASE_NAME, default=LOGS_PHASE_DEFAULT
)


def initialize_logs(
    debug_path: str = LOGS_DEBUG_FILE_PATH,
    info_path: str = LOGS_INFO_FILE_PATH,
    error_path: str = LOGS_ERROR_FILE_PATH,
    base_level: int = LOGS_FILE_BASE_LEVEL,
    max_bytes: int = LOGS_FILE_MAX_BYTES,
    backup_count: int = LOGS_FILE_BACKUP_COUNT,
    logs_format: str = LOGS_FORMAT,
) -> None:
    """
    Set up global logging configuration.

    This function configures logging to:
        - Write debug, info, and error messages to separate rotating files.
        - Use a consistent log format for all messages.

    Args:
        debug_path (str): Path for debug log file.
        info_path (str): Path for info log file.
        error_path (str): Path for error log file.
        base_level (int): Base logging level for the root logger.
        max_bytes (int): Maximum file size in bytes before rotation.
        backup_count (int): Number of backup files to keep.
        logs_format (str): Log message format.

    Returns:
        None
    """
    # Create file handlers for debug,
    # info, and error logs
    debug_file_handler = build_logs_file_handler(
        debug_path,
        logging.DEBUG,
        max_bytes,
        backup_count,
        logs_format,
    )
    info_file_handler = build_logs_file_handler(
        info_path,
        logging.INFO,
        max_bytes,
        backup_count,
        logs_format,
    )
    error_file_handler = build_logs_file_handler(
        error_path,
        logging.ERROR,
        max_bytes,
        backup_count,
        logs_format,
    )

    # Configure the root logger
    # with all handlers
    logging.basicConfig(
        level=base_level,
        handlers=[
            debug_file_handler,
            info_file_handler,
            error_file_handler,
        ],
    )
