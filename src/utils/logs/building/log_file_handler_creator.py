import logging
from logging.handlers import RotatingFileHandler

from pipeline.const import (
    LOGS_FILE_MAX_BYTES,
    LOGS_FILE_BACKUP_COUNT,
    LOGS_FORMAT,
)


def create_logs_file_handler(
    logs_path: str,
    logs_level: int,
    max_bytes: int = LOGS_FILE_MAX_BYTES,
    backup_count: int = LOGS_FILE_BACKUP_COUNT,
    logs_format: str = LOGS_FORMAT,
) -> RotatingFileHandler:
    """
    Set up a RotatingFileHandler for logging to a file.

    This function creates and configures a rotating file handler
    with the standard formatter and specified logs level. It is
    used to write logs to disk while limiting file size and
    keeping backups.

    Args:
        logs_path (str): Path to the logs file.
        logs_level (int): Logging level.
        max_bytes (int): Maximum file size in bytes before rotation.
        backup_count (int): Number of backup files to keep.
        logs_format (str): Format string for logs messages.

    Returns:
        RotatingFileHandler: Rotating file handler configured.
    """
    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        logs_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    # Set logs level
    file_handler.setLevel(logs_level)

    # Apply formatter
    file_handler.setFormatter(logging.Formatter(logs_format))

    return file_handler
